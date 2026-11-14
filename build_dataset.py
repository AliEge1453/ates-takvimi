"""
Uyumlulastirilmis (harmonized) hucre-gun veri seti uretici
==========================================================
FIRMS'ten (arsiv SP + yakin tarih NRT otomatik) MODIS (Terra, Aqua) ve VIIRS (SNPP, NOAA-20, NOAA-21)
tespitlerini yil yil indirir, kalici kaynaklari (sanayi flare'i vb.) ayiklar ve her yil icin
SEYREK bir hucre x gun tablosu yazar:

    dataset/<BOLGE>_<YIL>.csv.gz
      gun, lat_c, lon_c                      (hucre merkezi; varsayilan 0.25 derece)
      n_Terra, n_Aqua, n_VIIRS_SNPP, n_VIIRS_NOAA20, n_VIIRS_NOAA21     (tespit sayisi)
      frp_Terra, ... , frp_VIIRS_NOAA21                                 (FRP toplami, MW)
      n_statik                               (o hucre-gunde cikarilan kalici-kaynak tespiti)
    dataset/<BOLGE>_manifest.csv             (yil basina ozet + her uydunun medyan yerel gecis saati)

Sensor calismiyorsa (orn. NOAA-20 2018 oncesi) o sutunlar 0 olur. Indirmeler onbellekli: yarida kesilirse
ayni komutu tekrar calistir, kalan yerden devam eder.

Calistir (harmonize_v12.py ile AYNI klasorde, map_key.txt orada olmali):
    python build_dataset.py --region TR --years 2019-2026
    python build_dataset.py --region LA --years 2024-2025
    python build_dataset.py --region CA --years 2019-2026 --cell 0.25
Ayni bayraklar harmonize_v12.py ile ortaktir (--pause, --refresh, --bbox).
"""

import sys
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

import harmonize_v12 as h

GROUPS = ["Terra", "Aqua", "VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21"]
CELL = float(h._argval("--cell", 0.25))
YEARS = h._argval("--years", "2019-2026")
OUT = h.HERE / "dataset"
VIIRS_SENSORS = ("VIIRS_SNPP",) if h._argval("--viirs", "hepsi").lower() == "snpp" else ("VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21")
ATLAS = []            # VIIRS doneminde bulunan kalici kaynak noktalari (lat, lon); MODIS-tek yillara uygulanir


def atlas_path(region):
    return OUT / f"{region}_kalici_atlas.csv"


def load_atlas(region):
    p = atlas_path(region)
    if p.exists():
        a = pd.read_csv(p)
        if len(a):
            ATLAS.append(a[["lat", "lon"]].to_numpy(float))


def save_atlas(region):
    if not ATLAS:
        return
    a = pd.DataFrame(np.vstack(ATLAS), columns=["lat", "lon"]).round(3).drop_duplicates()
    OUT.mkdir(exist_ok=True)
    a.to_csv(atlas_path(region), index=False)


def modis_only_static(modis, n_days):
    """VIIRS olmayan yillarda (2003-2011) kalici kaynak bayragi:
    (1) VIIRS doneminin kalici kaynak atlasina 1 km'den yakin MODIS tespitleri,
    (2) ayni 1 km icinde yilin en az %10'u kadar farkli gunde tekrar eden tespitler."""
    from sklearn.neighbors import BallTree
    flag = np.zeros(len(modis), bool)
    if not len(modis):
        return flag
    Xm = np.radians(modis[["lat", "lon"]].to_numpy(float))
    r = 1.0 / h.EARTH_RADIUS_KM
    if ATLAS:
        A = pd.DataFrame(np.vstack(ATLAS), columns=["lat", "lon"]).round(3).drop_duplicates().to_numpy(float)
        dist, _ = BallTree(np.radians(A), metric="haversine").query(Xm, k=1)
        flag |= dist[:, 0] <= r
    dm = (pd.to_datetime(modis["timestamp"]).dt.normalize() - pd.to_datetime(modis["timestamp"]).min().normalize()).dt.days.to_numpy()
    idx = BallTree(Xm, metric="haversine").query_radius(Xm, r=r)
    nd = np.array([len(np.unique(dm[ii])) for ii in idx])
    flag |= nd >= max(h.STATIC_MIN_DAYS, int(np.ceil(0.10 * n_days)))
    return flag


def write_empty(region, year):
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(columns=["gun", "lat_c", "lon_c"] + [f"n_{g}" for g in GROUPS] + [f"frp_{g}" for g in GROUPS] + ["n_statik"]).to_csv(OUT / f"{region}_{year}.csv.gz", index=False, compression="gzip")


def parse_years(txt):
    if "-" in txt:
        a, b = txt.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in txt.split(",")]


def aggregate(det, cell):
    """Seyrek hucre x gun tablosu (sadece tespit olan hucre-gunler)."""
    if det.empty:
        return pd.DataFrame()
    d = det.copy()
    d["gun"] = pd.to_datetime(d["timestamp"]).dt.date.astype(str)
    d["frp_n"] = pd.to_numeric(d["frp"], errors="coerce").fillna(0.0)
    d["ci"] = np.floor(d["lat"] / cell).astype(int)
    d["cj"] = np.floor(d["lon"] / cell).astype(int)
    g = d.groupby(["gun", "ci", "cj", "grup"]).agg(n=("lat", "size"), frp=("frp_n", "sum"))
    n = g["n"].unstack("grup", fill_value=0).reindex(columns=GROUPS, fill_value=0).add_prefix("n_")
    f = g["frp"].unstack("grup", fill_value=0.0).reindex(columns=GROUPS, fill_value=0.0).add_prefix("frp_")
    return n.join(f)


def local_medians(modis, viirs):
    """Gunduz gecislerinin medyan yerel gunes saati (UTC saati + boylam/15)."""
    out = {}
    allp = pd.concat([modis.assign(grup=modis["satellite"].astype(str)),
                      viirs.assign(grup=viirs["source"].astype(str))], ignore_index=True)
    if allp.empty:
        return out
    ts = pd.to_datetime(allp["timestamp"])
    local = ((ts.dt.hour + ts.dt.minute / 60 + allp["lon"] / 15.0) % 24).to_numpy()
    day = ~allp["daynight"].astype(str).str.upper().eq("N").to_numpy()
    grp = allp["grup"].to_numpy()
    for g in GROUPS:
        sel = (grp == g) & day
        if sel.sum() >= 20:
            out[g] = float(np.median(local[sel]))
    return out


def process_year(year, key, avail, region):
    today = datetime.now(timezone.utc).date()
    start = date(year, 1, 1)
    end = min(date(year, 12, 31), today - timedelta(days=1))
    if end < start:
        return None
    modis = h.standardize(h.fetch_sensor("MODIS", key, start, end, avail), "MODIS")
    frames = []
    for sensor in VIIRS_SENSORS:
        raw = h.fetch_sensor(sensor, key, start, end, avail)
        if len(raw):
            frames.append(h.standardize(raw, sensor))
    viirs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=modis.columns)
    if modis.empty and viirs.empty:
        print(f"  {year}: veri yok (bos dosya yazildi)")
        write_empty(region, year)
        return None
    if (today - end).days <= 3 and len(viirs) and len(modis):
        modis, viirs = h.trim_incomplete(modis, viirs)

    # kalici kaynaklar: tum yil boyunca ayni yerde tekrar eden tespitler (iki sensorden de varsa)
    sm = np.zeros(len(modis), bool)
    sv = np.zeros(len(viirs), bool)
    if len(modis) and len(viirs):
        sm, sv, n_days, min_days, _, _ = h.flag_static(modis, viirs)
        if sv.any():
            ATLAS.append(viirs.loc[sv, ["lat", "lon"]].to_numpy(float))
    elif len(modis):
        sm = modis_only_static(modis, (end - start).days + 1)
    stat = pd.concat([modis[sm].assign(grup=modis.loc[sm, "satellite"].astype(str)),
                      viirs[sv].assign(grup=viirs.loc[sv, "source"].astype(str))], ignore_index=True)
    modis_c, viirs_c = modis[~sm].reset_index(drop=True), viirs[~sv].reset_index(drop=True)
    det = pd.concat([modis_c.assign(grup=modis_c["satellite"].astype(str)),
                     viirs_c.assign(grup=viirs_c["source"].astype(str))], ignore_index=True)
    table = aggregate(det, CELL)
    if table.empty:
        write_empty(region, year)
        return None
    st = aggregate(stat, CELL)
    table["n_statik"] = (st[[f"n_{g}" for g in GROUPS]].sum(axis=1).reindex(table.index).fillna(0).astype(int)
                         if len(st) else 0)
    table = table.reset_index()
    table["lat_c"] = (table["ci"] + 0.5) * CELL
    table["lon_c"] = (table["cj"] + 0.5) * CELL
    cols = ["gun", "lat_c", "lon_c"] + [f"n_{g}" for g in GROUPS] + [f"frp_{g}" for g in GROUPS] + ["n_statik"]
    table = table[cols]
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{region}_{year}.csv.gz"
    table.to_csv(path, index=False, compression="gzip")

    med = local_medians(modis_c, viirs_c)
    row = {"yil": year, "gun_sayisi": (end - start).days + 1, "hucre_gun_satiri": len(table),
           "statik_MODIS": int(sm.sum()), "statik_VIIRS": int(sv.sum())}
    for g in GROUPS:
        row[f"n_{g}"] = int(table[f"n_{g}"].sum())
    for g in GROUPS:
        row[f"yerel_saat_{g}"] = round(med[g], 2) if g in med else np.nan
    print(f"  {year}: {len(table)} hucre-gun satiri -> {path.name}  (MODIS {len(modis_c)}, VIIRS {len(viirs_c)}; "
          f"kalici cikarilan: {int(sm.sum())}+{int(sv.sum())})")
    return row


def main():
    key = h.get_map_key()
    region = h.REGION
    years = parse_years(YEARS)
    avail = h.get_availability(key)
    print(f"Bolge: {region}  kutu: {h.AREA_BBOX}  yillar: {years[0]}-{years[-1]}  hucre: {CELL} derece")
    rows = []
    t0 = time.time()
    load_atlas(region)
    # yeniden eskiye: VIIRS yillari once islenir, kalici kaynak atlasi MODIS-tek yillara (2003-2011) hazir olur
    for y in sorted(years, reverse=True):
        print(f"\n== {y} ==")
        try:
            r = process_year(y, key, avail, region)
        except Exception as e:
            if "MAP_KEY" in str(e):
                print(f"  [DUR] {y}: FIRMS anahtari calismiyor ({e}). Islem durduruldu; 20-30 dk sonra tekrar dene.")
                save_atlas(region)
                sys.exit(3)
            body = ""
            resp = getattr(e, "response", None)
            if resp is not None:
                body = " | FIRMS cevabi: " + (resp.text or "").strip()[:200]
            print(f"  [HATA] {y}: {str(e).replace(key, '<MAP_KEY>')}{body}  (ayni komutu tekrar calistir; indirilenler onbellekte)")
            continue
        if r:
            rows.append(r)
        print(f"  [{time.time() - t0:5.0f} sn]")
    save_atlas(region)
    if not rows:
        sys.exit("Hic yil uretilemedi.")
    man = pd.DataFrame(rows)
    old = OUT / f"{region}_manifest.csv"
    if old.exists():                       # baska yillarin satirlarini koru (toplu_indir eksik yillari tek tek isler)
        prev = pd.read_csv(old)
        man = pd.concat([prev[~prev["yil"].isin(man["yil"])], man], ignore_index=True)
    man = man.sort_values("yil")
    OUT.mkdir(exist_ok=True)
    man.to_csv(OUT / f"{region}_manifest.csv", index=False)
    print("\nOZET (yil basina tespit sayisi, kalici kaynak cikarildi; yerel_saat = gunduz gecisi medyani):")
    show = man[["yil", "hucre_gun_satiri", "statik_MODIS", "statik_VIIRS"] + [f"n_{g}" for g in GROUPS]
               + ["yerel_saat_Aqua", "yerel_saat_Terra", "yerel_saat_VIIRS_SNPP"]]
    print(show.to_string(index=False))
    print(f"\nKaydedildi: {OUT.name}/{region}_<yil>.csv.gz ve {region}_manifest.csv")


if __name__ == "__main__":
    main()
