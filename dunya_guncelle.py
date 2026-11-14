"""
Dunya veri setini bugune kadar gunceller (indirmeden sonraki gunler, NOAA-20/21 ile).

FIRMS S-NPP VIIRS uretimini 2 Kasim 2026 13:00 UTC'de durdurdu. Dunya veri seti S-NPP ile 1 Kasim'a kadar.
Bu betik o gunden bugune kadar olan kismi MODIS + NOAA-20 + NOAA-21 ile indirip mevcut karolara ekler.
Ocak-Ekim tekrar indirilmez; sadece eksik gunler (genelde 5-10 istek, birkac dakika).

    cd C:\\nasa_harmonization   (ya da proje klasoru neredeyse)
    python dunya_guncelle.py           # eksik gunleri indir, karolara ekle
    python dunya_guncelle.py --test    # sadece plani yaz, indirme yapma

Eklenen satirlarda n_VIIRS_ort sutunu var: harmonizasyonda kullanilan VIIRS degeri.
  * 2 Kasim'dan once: S-NPP (kalibrasyonla ayni uydu)
  * 2 Kasim ve sonrasi: o gun veri veren NOAA uydularinin ortalamasi
Eski satirlarda bu sutun bos; sunucu.py bossa n_VIIRS_SNPP'yi kullanir.
Tekrar calistirmak guvenli: onceki eklemeyi silip yeniden yazar. Sonra sunucuyu yeniden baslat.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

import dunya_indir as di
import harmonize_v12 as h

SNPP_END = date(2026, 11, 2)          # bu gunden itibaren S-NPP yok
GCODE = {"VIIRS_SNPP": 2, "VIIRS_NOAA20": 3, "VIIRS_NOAA21": 4}
NOAA = ["n_VIIRS_NOAA20", "n_VIIRS_NOAA21"]
say = di.say


def last_avail(avail, sensor, fallback):
    sp, nrt = h.SENSOR_IDS[sensor]
    return max([avail[x][1] for x in (sp, nrt) if x in avail], default=fallback)


def make_plan(start, end, avail):
    plan = []

    def add(sensor, a, b):
        cur = a
        while cur <= b:
            n = min(5, (b - cur).days + 1)
            for src, st, nn in h.plan_window(sensor, cur, n, avail):
                plan.append((sensor, src, st, nn))
            cur += timedelta(days=n)

    add("MODIS", start, end)
    if start < SNPP_END:
        add("VIIRS_SNPP", start, min(end, SNPP_END - timedelta(days=1)))
    if end >= SNPP_END:
        for s in ("VIIRS_NOAA20", "VIIRS_NOAA21"):
            add(s, max(start, SNPP_END), end)
    return plan


def build_table(year, parts, ndays):
    k = np.concatenate([p[0] for p in parts])
    d = np.concatenate([p[1] for p in parts])
    g = np.concatenate([p[2] for p in parts])
    lat_i = k // 36001 - 9000
    lon_i = k % 36001 - 18000
    # kalici kaynak: ~2 km kare, eklenen gunlerin belirli bir oraninda tespit (en az h.STATIC_MIN_DAYS gun).
    # Kisa pencerede (< 60 gun) %50 cok gevsek: 12 gunde 6 gun yanan gercek yangini da siler. Bu yuzden %75.
    k2 = (lat_i // 2 + 4500) * 18001 + (lon_i // 2 + 9000)
    pair = np.unique(k2 * 400 + d)
    kk, nd = np.unique(pair // 400, return_counts=True)
    frac = 0.75 if ndays < 60 else h.STATIC_FRACTION
    thr = max(h.STATIC_MIN_DAYS, int(np.ceil(frac * ndays)))
    static = np.isin(k2, kk[nd >= thr])
    cid = (np.floor_divide(lat_i, di.CELL_U) + 400) * 2000 + (np.floor_divide(lon_i, di.CELL_U) + 800)

    keep = ~static
    code = (cid[keep].astype(np.int64) * 400 + d[keep]) * 8 + g[keep]
    u, c = np.unique(code, return_counts=True)
    t = pd.DataFrame({"cd": u // 8, "g": u % 8, "n": c}).pivot_table(index="cd", columns="g", values="n",
                                                                       aggfunc="sum", fill_value=0)
    for j in range(5):
        if j not in t.columns:
            t[j] = 0
    scode = cid[static].astype(np.int64) * 400 + d[static]
    us, cs = np.unique(scode, return_counts=True)
    t["n_statik"] = pd.Series(cs, index=us).reindex(t.index).fillna(0).astype(int).to_numpy()

    idx = t.index.to_numpy()
    cell, day = idx // 400, (idx % 400).astype(int)
    ci, cj = cell // 2000 - 400, cell % 2000 - 800
    tab = pd.DataFrame({
        "gun": (pd.Timestamp(year, 1, 1) + pd.to_timedelta(day, unit="D")).strftime("%Y-%m-%d"),
        "lat_c": (ci + 0.5) * 0.25, "lon_c": (cj + 0.5) * 0.25,
        "n_Terra": t[0].to_numpy(), "n_Aqua": t[1].to_numpy(), "n_VIIRS_SNPP": t[2].to_numpy(),
        "n_VIIRS_NOAA20": t[3].to_numpy(), "n_VIIRS_NOAA21": t[4].to_numpy(), "n_statik": t["n_statik"].to_numpy()})

    # harmonizasyonda kullanilacak VIIRS degeri: gun gun aktif NOAA uydu sayisina bol (dunya genelinde tespit veren)
    after = pd.to_datetime(tab["gun"]).dt.date >= SNPP_END
    act = (tab[NOAA].groupby(tab["gun"]).transform("sum") > 0).sum(axis=1)
    tab["n_VIIRS_ort"] = np.where(after, tab[NOAA].sum(axis=1) / act.clip(lower=1), tab["n_VIIRS_SNPP"]).round(3)
    tab["tile"] = (np.floor(tab["lat_c"] / di.TILE).astype(int).astype(str) + "_" +
                   np.floor(tab["lon_c"] / di.TILE).astype(int).astype(str))
    return tab, int(len(k) - static.sum()), int(static.sum())


def main():
    key = h.get_map_key()
    avail = h.get_availability(key)
    today = datetime.now(timezone.utc).date()
    year = today.year
    man = di.manifest_load()
    yinfo = man.get("yillar", {}).get(str(year))
    if not yinfo:
        say(f"{year} dunya veri setinde yok. Once: python dunya_indir.py --yillar {year}-{year}")
        return
    base_end = date(year, 1, 1) + timedelta(days=int(yinfo["gun"]) - 1)
    start = base_end + timedelta(days=1)
    end = min(today - timedelta(days=1), last_avail(avail, "MODIS", today),
              last_avail(avail, "VIIRS_NOAA20" if today > SNPP_END else "VIIRS_SNPP", today))
    say(f"dunya veri seti {year}: ana kisim {base_end} tarihine kadar. Eklenecek: {start}..{end}")
    if end < start:
        say("Eklenecek yeni gun yok (FIRMS henuz yeni veri vermemis olabilir).")
        return
    plan = make_plan(start, end, avail)
    say(f"  {len(plan)} istek: " + ", ".join(f"{src} {st:%d.%m}+{n}" for _, src, st, n in plan))
    if "--test" in sys.argv:
        return

    def job(p):
        sensor, src, st, n = p
        (k, d, g), nb, sec = di.fetch_world(key, src, st, n)
        if sensor != "MODIS":
            g = np.full(len(k), GCODE[sensor], np.int8)
        return (k, d, g), nb, src, st

    parts, mb = [], 0.0
    with ThreadPoolExecutor(max_workers=di.WORKERS) as ex:
        for i, fu in enumerate(as_completed([ex.submit(job, p) for p in plan]), 1):
            arr, nb, src, st = fu.result()
            parts.append(arr)
            mb += nb / 1e6
            say(f"  {i}/{len(plan)} {src} {st:%d.%m}: {len(arr[0]):,} tespit")
    if not sum(len(p[0]) for p in parts):
        say("Hic tespit gelmedi; karolar degistirilmedi.")
        return

    tab, n_det, n_stat = build_table(year, parts, (end - start).days + 1)
    cols = di.COLS + ["n_VIIRS_ort"]
    new_by_tile = {t: p for t, p in tab.groupby("tile")}
    touched = 0
    for f in sorted(di.OUT.glob(f"T*_{year}.csv.gz")):
        tile = f.name[1:-len(f"_{year}.csv.gz")]
        old = pd.read_csv(f)
        old = old[pd.to_datetime(old["gun"]).dt.date <= base_end]       # onceki eklemeyi temizle
        if "n_VIIRS_ort" not in old.columns:
            old["n_VIIRS_ort"] = np.nan
        add = new_by_tile.pop(tile, None)
        out = old[cols] if add is None else pd.concat([old[cols], add[cols]], ignore_index=True)
        out.to_csv(f, index=False, compression="gzip")
        touched += add is not None
    for tile, add in new_by_tile.items():                                  # daha once hic tespiti olmayan karolar
        add[cols].to_csv(di.OUT / f"T{tile}_{year}.csv.gz", index=False, compression="gzip")
        touched += 1

    yinfo["ek"] = {"bas": str(start), "son": str(end), "hucre_gun": int(len(tab)), "tespit": n_det,
                   "kalici_ayiklanan": n_stat,
                   "kaynak": "MODIS NRT + VIIRS NOAA-20/21 NRT (S-NPP 2 Kasim 2026'da kapandi)",
                   "guncellendi": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    di.manifest_save(man)
    say(f"Tamam: {start}..{end}, {len(tab):,} hucre-gun, {n_det:,} tespit ({n_stat:,} kalici kaynak ayiklandi), "
        f"{touched} karo guncellendi, {mb:,.0f} MB indirildi.")
    say("Sunucuyu yeniden baslat: Ctrl+C, sonra python sunucu.py")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        say(f"DUR: {e}. 20-30 dk sonra ayni komutu calistir.")
        sys.exit(3)
