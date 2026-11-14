"""
Terra / Aqua yorunge kaymasi ve MODIS/VIIRS tespit oranlari - yil yil
======================================================================
Her yil icin AYNI mevsimden 30 gun (16 Eylul - 15 Ekim) indirir (gecmis yillar FIRMS arsiv SP,
2026 NRT), kalici kaynaklari (sanayi flare'i vb.) cikarir ve sunlari hesaplar:
  1) her uydunun medyan yerel gunes saati (yorunge kaymasi)
  2) gunduz gecisleri icin Aqua/SNPP ve Terra/SNPP tespit SAYISI ve FRP orani (gunler uzerinden
     bootstrap %95 guven araligi)
Soru: MODIS'in gecis saati kaydikca, VIIRS'e gore yakaladigi yangin orani degisiyor mu?

Calistir (harmonize_v12.py ile AYNI klasorde, map_key.txt orada olmali):
    python drift_history.py
    python drift_history.py --from 2019 --to 2026
"""

import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

import harmonize_v12 as h

FIRST_YEAR = int(h._argval("--from", 2019))
LAST_YEAR = int(h._argval("--to", 2026))
START = (9, 16)                       # her yil 16 Eylul'den baslayan 30 gun (6 pencere x 5 gun)
N_WINDOWS = 6
NRT_FROM_YEAR = 2026                  # bu yildan itibaren NRT, oncesi SP (arsiv)
SOURCES = {"sp": {"MODIS": "MODIS_SP", "SNPP": "VIIRS_SNPP_SP"},
           "nrt": {"MODIS": "MODIS_NRT", "SNPP": "VIIRS_SNPP_NRT"}}
BOOT = 500


def fetch_year(year, key):
    kind = "nrt" if year >= NRT_FROM_YEAR else "sp"
    out = {}
    for sensor in ("MODIS", "SNPP"):
        frames = []
        for k in range(N_WINDOWS):
            start = date(year, START[0], START[1]) + timedelta(days=5 * k)
            df, _ = h._fetch_window(SOURCES[kind][sensor], key, start, 5)
            if len(df):
                frames.append(df)
        if not frames:
            return None, kind
        out[sensor] = pd.concat(frames, ignore_index=True)
    return out, kind


def medians(modis, viirs):
    """(uydu, D/N) -> (medyan yerel gunes saati [saat], nokta sayisi). Gece gecisleri icin gece yarisi sarmasi duzeltilir."""
    allp = pd.concat([modis.assign(grup=modis["satellite"].astype(str)),
                      viirs.assign(grup="SNPP")], ignore_index=True)
    ts = pd.to_datetime(allp["timestamp"])
    local = ((ts.dt.hour + ts.dt.minute / 60 + allp["lon"] / 15.0) % 24).to_numpy()
    night = allp["daynight"].astype(str).str.upper().eq("N").to_numpy()
    local = np.where(night & (local < 12), local + 24, local)
    out = {}
    grp = allp["grup"].to_numpy()
    for g in np.unique(grp):
        for dn, is_n in (("D", False), ("N", True)):
            sel = (grp == g) & (night == is_n)
            if sel.sum() >= 20:
                out[(g, dn)] = (float(np.median(local[sel])), int(sel.sum()))
    return out


def per_day(df, mask, days):
    d = pd.to_datetime(df.loc[mask, "timestamp"]).dt.date
    cnt = d.value_counts().reindex(days, fill_value=0).to_numpy(float)
    frp = (pd.to_numeric(df.loc[mask, "frp"], errors="coerce").fillna(0.0)
           .groupby(d).sum().reindex(days, fill_value=0.0).to_numpy(float))
    return cnt, frp


def ratio_ci(num, den, seed=0):
    rng = np.random.default_rng(seed)
    n = len(num)
    idx = rng.integers(0, n, (BOOT, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = np.nanpercentile(r, [2.5, 97.5])
    return float(num.sum() / den.sum()), float(lo), float(hi)


def fmt_ci(t):
    return f"{t[0]:.2f} ({t[1]:.2f}-{t[2]:.2f})"


def main():
    key = h.get_map_key()
    rows, ratios = [], []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        try:
            raw, kind = fetch_year(year, key)
        except Exception as e:
            print(f"  {year}: indirilemedi ({e})")
            continue
        if raw is None:
            print(f"  {year}: veri bos ({kind})")
            continue
        modis = h.standardize(raw["MODIS"], "MODIS")
        viirs = h.standardize(raw["SNPP"], "VIIRS_SNPP")
        n0m, n0v = len(modis), len(viirs)
        sm, sv, *_ = h.flag_static(modis, viirs)
        modis, viirs = modis[~sm].reset_index(drop=True), viirs[~sv].reset_index(drop=True)
        print(f"  {year} ({kind.upper()}): MODIS {n0m} (kalici {int(sm.sum())}), VIIRS-SNPP {n0v} (kalici {int(sv.sum())})")

        med = medians(modis, viirs)
        for (g, dn), (hour, n) in med.items():
            rows.append({"yil": year, "uydu": f"{g} {dn}", "saat": hour, "n": n})

        days = [date(year, START[0], START[1]) + timedelta(days=i) for i in range(5 * N_WINDOWS)]
        is_d = modis["daynight"].astype(str).str.upper().eq("D")
        sat = modis["satellite"].astype(str)
        sn_c, sn_f = per_day(viirs, viirs["daynight"].astype(str).str.upper().eq("D"), days)
        rec = {"yil": year, "MODIS_kalici": int(sm.sum())}
        for g in ("Terra", "Aqua"):
            c, f = per_day(modis, is_d & sat.eq(g), days)
            rec[f"n_{g}"] = int(c.sum())
            rec[f"{g}/SNPP_sayi"] = ratio_ci(c, sn_c)
            rec[f"{g}/SNPP_FRP"] = ratio_ci(f, sn_f)
        rec["n_SNPP"] = int(sn_c.sum())
        ratios.append(rec)
    if not rows:
        sys.exit("Hic veri alinamadi.")

    df = pd.DataFrame(rows)
    piv = df.pivot(index="yil", columns="uydu", values="saat")
    fmt = lambda x: h._hhmm(x) if pd.notna(x) else "-"
    show = piv.map(fmt) if hasattr(piv, "map") else piv.applymap(fmt)
    print("\nMEDYAN YEREL GUNES SAATI (Turkiye, 16 Eylul-15 Ekim; nominal: Terra 10:30/22:30, Aqua-SNPP 13:30/01:30)")
    print("(Enlem etkisi: ~38N'de nominal 13:30 -> ~13:00 gorunur; SNPP referans alinabilir.)")
    print(show.to_string())
    off = pd.DataFrame({"Aqua-SNPP (sa)": piv["Aqua D"] - piv["SNPP D"], "Terra-SNPP (sa)": piv["Terra D"] - piv["SNPP D"]}).round(2)
    print("\nGUNDUZ GECIS SAATI FARKI (SNPP'ye gore, saat)")
    print(off.to_string())
    print("\nYILDAN YILA DEGISIM (dakika)")
    print((piv[["Aqua D", "Terra D", "SNPP D"]].diff() * 60).round(0).to_string())
    print("\nDOGRUSAL KAYMA HIZI (dk/yil; hiz artiyorsa dogrusal uyum guncel hizi kucumser):")
    for col in piv.columns:
        sub = piv[col].dropna()
        if len(sub) >= 4:
            slope = np.polyfit(sub.index.to_numpy(float), sub.to_numpy(float), 1)[0] * 60
            print(f"  {col:<10} {slope:+6.1f} dk/yil  ({len(sub)} yil)")

    if ratios:
        rt = pd.DataFrame(ratios).set_index("yil")
        cols = ["MODIS_kalici", "n_Terra", "n_Aqua", "n_SNPP"]
        out = rt[cols].copy()
        for c in ["Aqua/SNPP_sayi", "Terra/SNPP_sayi", "Aqua/SNPP_FRP", "Terra/SNPP_FRP"]:
            out[c] = rt[c].map(fmt_ci)
        print("\nGUNDUZ GECISLERI: MODIS/VIIRS-SNPP TESPIT ORANI, kalici kaynaklar cikarildi (gunler uzerinden bootstrap %95 GA)")
        print(out.to_string())
        point = pd.DataFrame({c: rt[c].map(lambda t: t[0]) for c in
                              ["Aqua/SNPP_sayi", "Terra/SNPP_sayi", "Aqua/SNPP_FRP", "Terra/SNPP_FRP"]})
        if len(point) >= 5:
            print("\nSpearman korelasyonu: ORAN ile (MODIS-SNPP gecis saati farki), yillar uzerinden (n kucuk, yol gosterici):")
            for sat_, c in (("Aqua", "Aqua/SNPP_sayi"), ("Terra", "Terra/SNPP_sayi"),
                            ("Aqua", "Aqua/SNPP_FRP"), ("Terra", "Terra/SNPP_FRP")):
                x = off[f"{sat_}-SNPP (sa)"].reindex(point.index)
                print(f"  {c:<18} rho = {x.corr(point[c], method='spearman'):+.2f}   (n={int(x.notna().sum())})")
        rt.drop(columns=[c for c in rt.columns if "/" in c]).join(
            point.add_prefix("oran_")).to_csv(h.HERE / "drift_ratios.csv")
    df.to_csv(h.HERE / "drift_history.csv", index=False)
    print("\nKaydedildi: drift_history.csv, drift_ratios.csv")


if __name__ == "__main__":
    main()
