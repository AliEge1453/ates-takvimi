"""
Dunya veri seti: tum ulkeler icin MODIS (Terra, Aqua) + VIIRS SNPP, 0.25 derece hucre-gun tablosu.
FIRMS kotasi istek basina sayildigi icin (alan buyuklugune bakmiyor) dunyanin tamami tek kutu olarak indirilir.

    cd C:\\nasa_harmonization
    python dunya_indir.py --test                 # 2-3 istek: boyut, sure, kota maliyeti ve tahmini toplam sure
    python dunya_indir.py                        # 2026'dan geriye 2015'e kadar, 3 paralel istek (kaldigi yerden devam eder)
    python dunya_indir.py --yillar 2020-2026

Cikti: dataset/DUNYA/T<enlem10>_<boylam10>_<yil>.csv.gz  (10x10 derecelik karolar) + dataset/DUNYA/manifest.json
Ara dosyalar data_cache/dunya_tmp/<yil>/ altinda; yil bitince silinir (--tut ile saklanir).
sunucu.py bu veri setini gorurse dunyanin her yerini aninda hesaplar.

Kalici kaynak (gaz bacasi, volkan, sanayi) ayiklama: ayni ~2 km karede yilin en az %50'si kadar gunde tespit.
"""

import json
import shutil
import sys
import time
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import harmonize_v12 as h

HERE = Path(__file__).resolve().parent
OUT = HERE / "dataset" / "DUNYA"
TMP = HERE / "data_cache" / "dunya_tmp"
API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
GRP = ["Terra", "Aqua", "VIIRS_SNPP"]
COLS = ["gun", "lat_c", "lon_c", "n_Terra", "n_Aqua", "n_VIIRS_SNPP", "n_VIIRS_NOAA20", "n_VIIRS_NOAA21", "n_statik"]
CELL_U = 25          # 0.25 derece = 25 x 0.01
WORKERS = int(h._argval("--paralel", 3))
TILE = 10            # karo boyu (derece)


def argval(flag, default):
    return h._argval(flag, default)


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- indirme
def fetch_world(key, src, start, days, timeout=600):
    """Bir pencere (dunya). Dondurur: (key01 int64, gun int16, grup int8), bayt, saniye."""
    url = f"{API}/{key}/{src}/world/{days}/{start:%Y-%m-%d}"
    t0 = time.time()
    waits = kwaits = 0
    for attempt in range(1, 7):
        try:
            h._quota_guard(key)
            r = requests.get(url, timeout=timeout)
            if r.status_code == 400:
                body = (r.text or "").strip()
                if "invalid" in body.lower() and "key" in body.lower():
                    if kwaits < 2 and h._key_blocked_wait(key, body):
                        kwaits += 1
                        continue
                    raise RuntimeError(f"MAP_KEY calismiyor: {body[:100]}")
                if waits < 15:
                    waits += 1
                    h._wait_for_quota(key, body)
                    continue
            r.raise_for_status()
            text = r.text
            nbytes = len(text)
            if not text.strip():
                return (np.zeros(0, np.int64), np.zeros(0, np.int16), np.zeros(0, np.int8)), nbytes, time.time() - t0
            if "<html" in text[:300].lower():
                raise RuntimeError(f"FIRMS HTML dondu ({src} {start})")
            df = pd.read_csv(StringIO(text), usecols=lambda c: c in ("latitude", "longitude", "acq_date", "satellite"))
            del text
            lat_i = np.round(df["latitude"].to_numpy(float) * 100).astype(np.int64)
            lon_i = np.round(df["longitude"].to_numpy(float) * 100).astype(np.int64)
            k = (lat_i + 9000) * 36001 + (lon_i + 18000)
            d = pd.to_datetime(df["acq_date"]).dt.dayofyear.to_numpy().astype(np.int16) - 1
            if src.startswith("MODIS"):
                sat = df["satellite"].astype(str).str[0].str.upper().to_numpy()
                g = np.where(sat == "T", 0, 1).astype(np.int8)
            else:
                g = np.full(len(df), 2, np.int8)
            return (k, d, g), nbytes, time.time() - t0
        except RuntimeError:
            raise
        except Exception as e:
            wait = min(10 * attempt, 60)
            say(f"  [bilgi] {src} {start}: {type(e).__name__} {str(e)[:80]}; {wait} sn sonra tekrar ({attempt}/6)")
            time.sleep(wait)
    raise RuntimeError(f"{src} {start} indirilemedi")


def plan_year(year, avail, today):
    start = date(year, 1, 1)
    end = min(date(year, 12, 31), today - timedelta(days=1))
    if year == today.year:                      # son ortak gun: iki sensor de veri vermis olmali
        last = []
        for s in ("MODIS", "VIIRS_SNPP"):
            sp, nrt = h.SENSOR_IDS[s]
            m = max([avail[x][1] for x in (sp, nrt) if x in avail], default=end)
            last.append(m)
        end = min(end, min(last) - timedelta(days=1))
    plan = []
    for sensor in ("MODIS", "VIIRS_SNPP"):
        cur = start
        while cur <= end:
            n = min(5, (end - cur).days + 1)
            plan.extend(h.plan_window(sensor, cur, n, avail))
            cur += timedelta(days=n)
    return plan, start, end


# ---------------------------------------------------------------- isleme
def process_year(year, parts, ndays):
    k = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, np.int64)
    d = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, np.int16)
    g = np.concatenate([p[2] for p in parts]) if parts else np.zeros(0, np.int8)
    if not len(k):
        return {}
    lat_i = k // 36001 - 9000
    lon_i = k % 36001 - 18000
    # kalici kaynak: ~2 km kare (0.02 derece), yilin >= %50'si kadar gun
    k2 = (lat_i // 2 + 4500) * 18001 + (lon_i // 2 + 9000)
    pair = np.unique(k2 * 400 + d)
    kk, nd = np.unique(pair // 400, return_counts=True)
    thr = max(h.STATIC_MIN_DAYS, int(np.ceil(h.STATIC_FRACTION * ndays)))
    static = np.isin(k2, kk[nd >= thr])
    ci = np.floor_divide(lat_i, CELL_U)
    cj = np.floor_divide(lon_i, CELL_U)
    cid = (ci + 400) * 2000 + (cj + 800)                       # hucre kimligi
    out = {}
    for name, m, gsel in [("keep", ~static, None), ("stat", static, None)]:
        code = (cid[m].astype(np.int64) * 400 + d[m]) * 4 + g[m]
        u, c = np.unique(code, return_counts=True)
        out[name] = (u, c)
    u, c = out["keep"]
    gg = u % 4
    cd = u // 4
    t = pd.DataFrame({"cd": cd, "g": gg, "n": c}).pivot_table(index="cd", columns="g", values="n", aggfunc="sum", fill_value=0)
    for j in range(3):
        if j not in t.columns:
            t[j] = 0
    us, cs = out["stat"]
    st = pd.Series(cs, index=us // 4).groupby(level=0).sum()
    t["n_statik"] = st.reindex(t.index).fillna(0).astype(int)
    idx = t.index.to_numpy()
    cell = idx // 400
    day = (idx % 400).astype(int)
    ci = cell // 2000 - 400
    cj = cell % 2000 - 800
    tab = pd.DataFrame({
        "gun": (pd.Timestamp(year, 1, 1) + pd.to_timedelta(day, unit="D")).strftime("%Y-%m-%d"),
        "lat_c": (ci + 0.5) * 0.25, "lon_c": (cj + 0.5) * 0.25,
        "n_Terra": t[0].to_numpy(), "n_Aqua": t[1].to_numpy(), "n_VIIRS_SNPP": t[2].to_numpy(),
        "n_VIIRS_NOAA20": 0, "n_VIIRS_NOAA21": 0, "n_statik": t["n_statik"].to_numpy()})
    tab["tile"] = (np.floor(tab["lat_c"] / TILE).astype(int).astype(str) + "_" +
                   np.floor(tab["lon_c"] / TILE).astype(int).astype(str))
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob(f"T*_{year}.csv.gz"):
        old.unlink()
    for tile, part in tab.groupby("tile"):
        part[COLS].to_csv(OUT / f"T{tile}_{year}.csv.gz", index=False, compression="gzip")
    return {"yil": year, "gun": ndays, "hucre_gun": len(tab), "karo": int(tab["tile"].nunique()),
            "tespit": int(len(k) - static.sum()), "kalici_ayiklanan": int(static.sum()),
            "n_Terra": int(tab["n_Terra"].sum()), "n_Aqua": int(tab["n_Aqua"].sum()), "n_VIIRS_SNPP": int(tab["n_VIIRS_SNPP"].sum())}


def manifest_load():
    p = OUT / "manifest.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else {"yillar": {}}


def manifest_save(m):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), "utf-8")


# ---------------------------------------------------------------- calistirma
def test(key, avail):
    say("TEST: dunya icin 3 istek (MODIS 5 gun, SNPP 1 gun, SNPP 5 gun), Agustos 2024")
    q0 = h._quota_status(key)
    rows = []
    for src, st, n in [("MODIS_SP", date(2024, 8, 1), 5), ("VIIRS_SNPP_SP", date(2024, 8, 6), 1), ("VIIRS_SNPP_SP", date(2024, 8, 7), 5)]:
        (k, d, g), nb, sec = fetch_world(key, src, st, n)
        rows.append((src, n, len(k), nb / 1e6, sec))
        say(f"  {src:14s} {n} gun: {len(k):>9,} tespit, {nb / 1e6:6.1f} MB, {sec:5.1f} sn")
    time.sleep(5)
    q1 = h._quota_status(key)
    cost = None
    if q0 and q1:
        cost = max(1, (q1[0] - q0[0]) / 3)
        say(f"  kota: {q0[0]} -> {q1[0]} / {q1[1]}  (istek basina ~{cost:.0f} islem)")
    years = 12
    n_req = years * 146
    mb = years * 73 * (rows[0][3] + rows[2][3])
    sec_dl = years * 73 * (rows[0][4] + rows[2][4]) / WORKERS
    per10 = (0.65 * q1[1] / cost) if (cost and q1) else 300
    sec_quota = n_req / per10 * 600
    say(f"TAHMIN 2015-2026: {n_req} istek, ~{mb / 1000:.0f} GB indirme (diske yazilan ~{mb / 1000 / 8:.1f} GB ara dosya, yil bitince silinir),")
    say(f"   indirme suresi ~{sec_dl / 3600:.1f} saat, kota siniri ~{sec_quota / 3600:.1f} saat -> toplam ~{max(sec_dl, sec_quota) / 3600:.1f} saat")


def main():
    key = h.get_map_key()
    avail = h.get_availability(key)
    if "--test" in sys.argv:
        return test(key, avail)
    y0, y1 = map(int, argval("--yillar", "2015-2026").split("-"))
    today = datetime.now(timezone.utc).date()
    man = manifest_load()
    t_all = time.time()
    for year in range(y1, y0 - 1, -1):
        if str(year) in man["yillar"] and not (year == today.year and "--guncelle" in sys.argv):
            say(f"{year}: zaten var, atlaniyor")
            continue
        plan, start, end = plan_year(year, avail, today)
        ydir = TMP / str(year)
        ydir.mkdir(parents=True, exist_ok=True)
        say(f"== {year} ({start}..{end}): {len(plan)} istek ==")
        parts, t0, mb, done = [], time.time(), 0.0, 0
        todo = []
        for src, st, n in plan:
            f = ydir / f"{src}_{st:%Y%m%d}_{n}.npz"
            if f.exists() and (src.endswith("_SP") or "--yenile" not in sys.argv):
                z = np.load(f)
                parts.append((z["k"], z["d"], z["g"]))
            else:
                todo.append((src, st, n, f))
        if len(parts):
            say(f"  {len(parts)} pencere onceki calismadan hazir, {len(todo)} indirilecek")

        def job(t):
            src, st, n, f = t
            (k, d, g), nb, sec = fetch_world(key, src, st, n)
            np.savez_compressed(f, k=k, d=d, g=g)
            return (k, d, g), nb

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(job, t) for t in todo]
            for fu in as_completed(futs):
                arr, nb = fu.result()
                parts.append(arr)
                mb += nb / 1e6
                done += 1
                if done % 10 == 0 or done == len(todo):
                    el = time.time() - t0
                    left = (len(todo) - done) * el / done / 60
                    say(f"  {done}/{len(todo)} istek, {mb:,.0f} MB, {el / 60:.0f} dk gecti, bu yil icin ~{left:.0f} dk kaldi")
        res = process_year(year, parts, (end - start).days + 1)
        parts.clear()
        if res:
            man["yillar"][str(year)] = res
            manifest_save(man)
            say(f"  {year} tamam: {res['hucre_gun']:,} hucre-gun, {res['karo']} karo, {res['tespit']:,} tespit "
                f"({res['kalici_ayiklanan']:,} kalici kaynak ayiklandi), {(time.time() - t0) / 60:.0f} dk")
        if "--tut" not in sys.argv:
            shutil.rmtree(ydir, ignore_errors=True)
    say(f"Bitti ({(time.time() - t_all) / 3600:.1f} saat). Sunucuyu yeniden baslat: python sunucu.py")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        say(f"DUR: {e}. 20-30 dk sonra ayni komutu calistir; inenler korunur.")
        sys.exit(3)
