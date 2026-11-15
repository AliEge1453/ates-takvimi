"""
Ates Takvimi - uygulama uretici
===============================
harmonize_v13.py, drift_history.py (ve varsa build_dataset.py) ciktilarini okur ve tek dosyalik,
sunucusuz bir web uygulamasi yazar:  app/ates_takvimi_<BOLGE>.html

Calistir (C:\\nasa_harmonization icinde):
    python build_app.py

dataset/ klasorundeki TUM bolgeler (build_dataset.py ciktilari) uygulamaya gomulur; kullanici bolge, yil
araligi ve haritada alan secer. Ayrica kullanici kendi FIRMS CSV dosyalarini yukleyebilir.

Gerekenler: cikti_<BOLGE>/grid_cell_day.csv, transfer_curve.csv, uyum_merdiveni.csv, kalicilik.csv
Istege bagli: drift_history.csv, drift_ratios.csv (drift_history.py), dataset/<BOLGE>_<yil>.csv.gz (build_dataset.py)
"""

import base64
import glob
import gzip
import json
import zlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def argval(flag, default):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


REGION = argval("--region", "TR").upper()
CELL = 0.25
OUT = HERE / "cikti_" / "x"  # placeholder, set below
OUT = HERE / f"cikti_{REGION}"
CITIES = {
    "TR": [("İstanbul", 41.01, 28.98), ("Ankara", 39.93, 32.86), ("İzmir", 38.42, 27.14), ("Konya", 37.87, 32.48),
           ("Adana", 37.00, 35.32), ("Diyarbakır", 37.91, 40.24), ("Şanlıurfa", 37.16, 38.79), ("Erzurum", 39.90, 41.27),
           ("Samsun", 41.29, 36.33), ("Edirne", 41.68, 26.56), ("Musul", 36.34, 43.13), ("Erbil", 36.19, 44.01)],
    "LA": [("Los Angeles", 34.05, -118.24), ("Malibu", 34.03, -118.78), ("Pasadena", 34.15, -118.14)],
    "NSW": [("Sydney", -33.87, 151.21), ("Canberra", -35.28, 149.13), ("Newcastle", -32.93, 151.78), ("Dubbo", -32.25, 148.60)],
    "CA": [("San Francisco", 37.77, -122.42), ("Los Angeles", 34.05, -118.24), ("Sacramento", 38.58, -121.49),
           ("Fresno", 36.74, -119.79), ("San Diego", 32.72, -117.16)],
}
BBOX = {"TR": "25–45°D · 35–43°K", "CA": "124.6–114.0°B · 32.4–42.1°K", "LA": "119.0–117.4°B · 33.6–34.6°K",
        "NSW": "140.9–153.7°D · 37.6–28.0°G", "PANTANAL": "59.0–54.5°B · 22.0–15.0°G", "GREECE": "19.3–28.4°D · 34.8–41.8°K",
        "PORTUGAL": "9.6–6.1°B · 36.9–42.2°K"}

PRESETS = {"TR", "LA", "NSW"}
try:
    from bolgeler import HEPSI as _HEPSI
except ImportError:
    _HEPSI = {}
REGION_NAME = {"TR": "Türkiye ve çevresi", "LA": "Los Angeles", "CA": "Kaliforniya", "NSW": "Yeni Güney Galler",
               "PANTANAL": "Pantanal", "GREECE": "Yunanistan", "PORTUGAL": "Portekiz"}
REGION_NAME.update({k: v["ad"] for k, v in _HEPSI.items() if k not in ("TR", "LA", "NSW")})


def pretty_level(s):
    """'3 hucre-gun 0.25 derece, +-1 gun' -> ['0.25° hücre (~25 km)', '±1 gün']"""
    s = re.sub(r"^\d+\s+", "", str(s))
    m = re.match(r"piksel \((.+?), \+-(\d+) sa\)", s)
    if m:
        kind = "ayak izi" if m.group(1) == "ayak izi" else m.group(1)
        return [f"Piksel · {kind}", f"±{m.group(2)} saat"]
    m = re.match(r"hucre-gun ([\d.]+) derece(, \+-1 gun)?", s)
    if m:
        km = {"0.25": "~25 km", "0.5": "~50 km", "1": "~100 km"}.get(m.group(1), "")
        return [f"{m.group(1)}° hücre ({km})", "±1 gün" if m.group(2) else "aynı gün"]
    return [s, ""]


def pretty_corr(s):
    m = re.match(r"mekansal desen \(([\d.]+) derece", s)
    if m:
        return f"Mekânsal desen · {m.group(1)}° hücre"
    m = re.match(r"hucre-hafta \(([\d.]+) derece", s)
    if m:
        return f"Hücre × hafta · {m.group(1)}°"
    if s.startswith("bolge gunluk"):
        return "Bölgesel günlük seyir"
    return s


def clean(seq):
    return [None if (x is None or (isinstance(x, float) and np.isnan(x))) else x for x in seq]


def need(path):
    if not path.exists():
        sys.exit(f"Bulunamadi: {path}\nOnce: python harmonize_v13.py --region {REGION} ...")
    return path


def transfer(v, curve, col):
    """VIIRS uydu-ortalamasi hucre-gun sayisindan beklenen MODIS sayisi (log-log enterpolasyon, V=0 -> 0)."""
    x = np.log(curve["VIIRS_ort_sayi"].to_numpy())
    y = np.log(np.clip(curve[col].to_numpy(), 1e-6, None))
    v = np.asarray(v, float)
    out = np.zeros_like(v)
    pos = v > 0
    lv = np.log(v[pos])
    slope_hi = (y[-1] - y[-2]) / (x[-1] - x[-2])
    slope_lo = (y[1] - y[0]) / (x[1] - x[0])
    est = np.interp(lv, x, y)
    est = np.where(lv > x[-1], y[-1] + slope_hi * (lv - x[-1]), est)
    est = np.where(lv < x[0], y[0] + slope_lo * (lv - x[0]), est)
    out[pos] = np.exp(est)
    return out


def comb_curve(curve):
    x = curve["VIIRS_ort_sayi"].to_numpy(float)
    y = (curve["Aqua_beklenen_sayi"] + curve["Terra_beklenen_sayi"]).to_numpy(float)
    return x, y


def f_comb(v, x, y):
    lx, ly = np.log(x), np.log(np.clip(y, 1e-6, None))
    v = np.asarray(v, float)
    out = np.zeros_like(v)
    pos = v > 0
    lv = np.log(v[pos])
    s_hi = (ly[-1] - ly[-2]) / (lx[-1] - lx[-2])
    s_lo = (ly[1] - ly[0]) / (lx[1] - lx[0])
    est = np.interp(lv, lx, ly)
    est = np.where(lv > lx[-1], ly[-1] + s_hi * (lv - lx[-1]), est)
    est = np.where(lv < lx[0], ly[0] + s_lo * (lv - lx[0]), est)
    out[pos] = np.exp(est)
    return out


DAY0 = pd.Timestamp("2000-01-01")


NOMINAL_LAST = 2020      # Terra ve Aqua'nin yorungeleri bu yila kadar sabit (2021 sonrasi kayma)


def split_years(ov):
    """Egitim/test yillari. Sabit yorunge yillari (<= 2020) yeterliyse: ilk 2/3 egitim, kalan sabit yillar test;
    kayma yillari (>= 2021) ayri raporlanir. Yetersizse eski kural: ortak yillarin ilk %60'i egitim."""
    nom = [y for y in ov if y <= NOMINAL_LAST]
    if len(nom) >= 3:
        ntr = max(2, int(round(len(nom) * 2 / 3)))
        return nom[:ntr], nom[ntr:], [y for y in ov if y > NOMINAL_LAST]
    ntr = max(1, int(round(len(ov) * 0.6)))
    return ov[:ntr], ov[ntr:], []


def yr_label(ys):
    return "" if not ys else (str(ys[0]) if len(ys) == 1 else f"{ys[0]}–{ys[-1]}")


def calibrate(df, k_default):
    """k: MODIS toplami / f(VIIRS) toplami egitim yillarinda; test yillarinda toplam isabet."""
    yrs = df["yil"]
    ov = sorted(y for y in yrs.unique() if df.loc[yrs == y, "f"].sum() > 0 and df.loc[yrs == y, "M"].sum() > 0)
    if not ov:
        return k_default, "bu bölgede MODIS ve VIIRS'in birlikte olduğu dönem yok; varsayılan katsayı.", None
    if len(ov) >= 3:
        a, t, _ = split_years(ov)
        tr, te = yrs.isin(a), yrs.isin(t)
        lab_tr, lab_te = yr_label(a), yr_label(t)
    else:
        sub = df[yrs.isin(ov)]
        days = np.sort(sub["d"].unique())
        cut = days[int(len(days) * 0.6)]
        tr, te = yrs.isin(ov) & (df["d"] < cut), yrs.isin(ov) & (df["d"] >= cut)
        lab_tr = f"{(DAY0 + pd.Timedelta(days=int(days[0]))).date()} – {(DAY0 + pd.Timedelta(days=int(cut) - 1)).date()}"
        lab_te = f"{(DAY0 + pd.Timedelta(days=int(cut))).date()} sonrası"
    k = float(df.loc[tr, "M"].sum() / max(df.loc[tr, "f"].sum(), 1e-9))
    ratio = float(k * df.loc[te, "f"].sum() / max(df.loc[te, "M"].sum(), 1)) if te.any() else None
    txt = f"eğitim {lab_tr}" + (f", test {lab_te}" if lab_te else "") + (f"; test döneminde toplam isabet %{100 * ratio:.0f}" if ratio else "")
    return k, txt, ratio


def pack(obj):
    return base64.b64encode(gzip.compress(json.dumps(obj, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")


EDGES = [0, 0.4, 0.75, 1.1, 1.6, 2.2, 3.2, 4.5, 6.5, 9, 13, 18, 26, 36, 52, 75, 105, 150, 220, 320, 1e9]


def fit_curve(rows, min_n=20):
    """Cok yillik egitim verisinden bolgeye ozgu donusum egrisi: VIIRS uydu-ortalamasi -> beklenen MODIS (Terra+Aqua).
    Logaritmik kutularda ortalama; monoton; son kutudan sonrasi dogrusal (log-log egim 1)."""
    r = rows[rows["V"] > 0]
    b = pd.cut(r["V"], EDGES)
    t = r.groupby(b, observed=True).agg(n=("V", "size"), x=("V", "mean"), y=("M", "mean"))
    t = t[t["n"] >= min_n]
    if len(t) < 4:
        return None
    x = t["x"].to_numpy(float)
    y = np.maximum.accumulate(np.clip(t["y"].to_numpy(float), 1e-4, None))
    return x, y


def f_curve(v, x, y, s_hi=None):
    lx, ly = np.log(x), np.log(np.clip(y, 1e-6, None))
    v = np.asarray(v, float)
    out = np.zeros_like(v)
    pos = v > 0
    lv = np.log(v[pos])
    if s_hi is None:
        s_hi = (ly[-1] - ly[-2]) / (lx[-1] - lx[-2])
    s_lo = (ly[1] - ly[0]) / (lx[1] - lx[0])
    est = np.interp(lv, lx, ly)
    est = np.where(lv > lx[-1], ly[-1] + s_hi * (lv - lx[-1]), est)
    est = np.where(lv < lx[0], ly[0] + s_lo * (lv - lx[0]), est)
    out[pos] = np.exp(est)
    return out


def overlap_years(rows):
    yrs = rows["yil"]
    return sorted(y for y in yrs.unique() if rows.loc[yrs == y, "V"].sum() > 0 and rows.loc[yrs == y, "M"].sum() > 0)


def year_unc(rows, k):
    """Yillik belirsizlik: sabit yorunge ortak yillarinda log(k*f/M) yillik sapmalarinin standart sapmasi."""
    ov = [y_ for y_ in overlap_years(rows) if y_ <= NOMINAL_LAST]
    if len(ov) < 3:
        return None
    yy = rows[rows["yil"].isin(ov)].groupby("yil").agg(M=("M", "sum"), f=("f", "sum"))
    yy = yy[yy["M"] > 0]
    if len(yy) < 3:
        return None
    return round(float(np.std(np.log(k * yy["f"] / yy["M"]), ddof=1)), 4)


def region_payload(key, rows, cell, x, y, k_default, note=""):
    """rows: lat, lon, d (gun indeksi), M, V  ->  sikistirilmis bolge nesnesi"""
    rows = rows[(rows["M"] > 0) | (rows["V"] > 0)].copy()
    rows["yil"] = (DAY0 + pd.to_timedelta(rows["d"], unit="D")).dt.year
    curve, s_hi, how = None, None, "30 günlük Türkiye eğrisi"
    ov = overlap_years(rows)
    a_, t_, dr_ = split_years(ov) if len(ov) >= 3 else ([], [], [])
    if a_:
        fit = fit_curve(rows[rows["yil"].isin(a_)])
        if fit is not None:
            x, y = fit
            s_hi, how = 1.0, "bölgeye özgü çok yıllı eğri"
            curve = {"x": np.round(x, 4).tolist(), "y": np.round(y, 5).tolist(), "hi": 1.0}
    rows["f"] = f_curve(rows["V"], x, y, s_hi)
    k, calib, ratio = calibrate(rows, k_default)
    extra = ""
    if t_:
        te = rows["yil"].isin(t_)
        dd = rows[te].groupby("d").agg(M=("M", "sum"), f=("f", "sum"))
        r_d = float(np.corrcoef(np.log1p(dd["M"]), np.log1p(k * dd["f"]))[0, 1])
        extra = f"; test günlük seyir r={r_d:.2f}"
    if dr_:
        yy = rows[rows["yil"].isin(dr_)].groupby("yil").agg(M=("M", "sum"), f=("f", "sum"))
        yr = (yy["M"] / (k * yy["f"]).clip(lower=1e-9) * 100).round().astype(int)
        extra += ("; yörünge kayması yıllarında ham MODIS ÷ uyumlu seri: " +
                  ", ".join(f"{a} %{b}" for a, b in yr.items()))
    calib = f"{how}; {calib}{extra}"
    cells = rows[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    cix = {(a, b): i for i, (a, b) in enumerate(zip(cells["lat"], cells["lon"]))}
    obj = {"key": key, "name": REGION_NAME.get(key, key), "cell": cell, "k": round(k, 4), "calib": calib, "note": note,
           "cities": [{"n": n, "lat": la, "lon": lo} for n, la, lo in CITIES.get(key, [])],
           "cells": {"lat": cells["lat"].round(3).tolist(), "lon": cells["lon"].round(3).tolist()},
           "c": [cix[(a, b)] for a, b in zip(rows["lat"], rows["lon"])], "d": rows["d"].astype(int).tolist(),
           "m": rows["M"].astype(int).tolist(), "v": np.round(rows["V"] * 10).astype(int).tolist()}
    if curve:
        obj["curve"] = curve
    u = year_unc(rows, k)
    if u:
        obj["unc"] = u
    b64 = pack(obj)
    yrs = sorted(rows["yil"].unique())
    print(f"  {key}: {len(rows):,} hücre-gün, {yrs[0]}–{yrs[-1]}, k={k:.2f} ({calib}), {len(b64) / 1e6:.1f} MB")
    return {"key": key, "name": obj["name"], "b64": b64, "curve": curve, "k": k}


def region_summary(key, rows, x, y, k_default):
    """Kanit tablosu icin: bolgenin kalibrasyon ve test ozeti (veri gommeden)."""
    rows = rows[(rows["M"] > 0) | (rows["V"] > 0)].copy()
    rows["yil"] = (DAY0 + pd.to_timedelta(rows["d"], unit="D")).dt.year
    ov = overlap_years(rows)
    a_, t_, dr_ = split_years(ov) if len(ov) >= 3 else ([], [], [])
    s_hi = None
    if a_:
        fit = fit_curve(rows[rows["yil"].isin(a_)], min_n=12)
        if fit is not None:
            x, y = fit
            s_hi = 1.0
    rows["f"] = f_curve(rows["V"], x, y, s_hi)
    k, calib, ratio = calibrate(rows, k_default)
    rd = rw = None
    if t_:
        te = rows["yil"].isin(t_)
        dd = rows[te].groupby("d").agg(M=("M", "sum"), f=("f", "sum"))
        if len(dd) > 2:
            rd = float(np.corrcoef(np.log1p(dd["M"]), np.log1p(k * dd["f"]))[0, 1])
        ww = rows[te].assign(w=rows["d"] // 7).groupby("w").agg(M=("M", "sum"), f=("f", "sum"))
        if len(ww) > 4:
            rw = float(np.corrcoef(np.log1p(ww["M"]), np.log1p(k * ww["f"]))[0, 1])
    unc = year_unc(rows, k)
    n_tr = int(rows.loc[rows["yil"].isin(a_), "M"].sum()) if a_ else int(rows["M"].sum())
    guven = ("yüksek" if unc is not None and unc < 0.12 and n_tr >= 3000 else
             "orta" if unc is not None and unc < 0.25 and n_tr >= 800 else "düşük")
    yv = rows.groupby("yil")["V"].sum()
    yv = yv[yv.index >= 2012]
    big = None
    if len(yv) >= 3 and yv.median() > 0:
        top = int(yv.idxmax())
        big = {"y": top, "kat": round(float(yv.max() / yv.median()), 1)}
    drift = None
    if dr_:
        yy = rows[rows["yil"].isin(dr_)].groupby("yil").agg(M=("M", "sum"), f=("f", "sum"))
        drift = {str(a): int(round(100 * m / max(k * f, 1e-9))) for a, m, f in zip(yy.index, yy["M"], yy["f"])}
    yrs = sorted(rows["yil"].unique())
    return {"key": key, "ad": REGION_NAME.get(key, key), "y0": int(yrs[0]), "y1": int(yrs[-1]),
            "egitim": f"{a_[0]}–{a_[-1]}" if a_ else "", "test": f"{t_[0]}–{t_[-1]}" if len(t_) > 1 else (str(t_[0]) if t_ else ""),
            "isabet": round(100 * ratio) if ratio else None, "rd": round(rd, 2) if rd is not None else None,
            "rw": round(rw, 2) if rw is not None else None, "guven": guven, "unc": unc, "ntr": n_tr, "buyuk": big, "kayma": drift}


EVIDENCE = []


def dataset_regions(x, y, k_default):
    out = []
    files = glob.glob(str(HERE / "dataset" / "*_[0-9][0-9][0-9][0-9].csv.gz"))
    keys = sorted({Path(f).name.rsplit("_", 1)[0] for f in files}, key=lambda k: (k != "TR", k))
    vcols = ["n_VIIRS_SNPP", "n_VIIRS_NOAA20", "n_VIIRS_NOAA21"]
    for key in keys:
        parts = []
        for f in sorted(glob.glob(str(HERE / "dataset" / f"{key}_[0-9][0-9][0-9][0-9].csv.gz"))):
            d = pd.read_csv(f)
            if d.empty:
                continue
            active = [c for c in vcols if c in d.columns and d[c].sum() > 0]        # o yil calisan VIIRS uydulari
            V = d[active].sum(axis=1) / len(active) if active else 0.0
            parts.append(pd.DataFrame({"lat": d["lat_c"], "lon": d["lon_c"], "d": (pd.to_datetime(d["gun"]) - DAY0).dt.days,
                                       "M": d["n_Terra"] + d["n_Aqua"], "V": V}))
        if parts:
            rows = pd.concat(parts, ignore_index=True)
            try:
                EVIDENCE.append(region_summary(key, rows, x, y, k_default))
            except Exception as e:
                print(f"  [bilgi] {key} özeti çıkarılamadı: {e}")
            if key in PRESETS or "--hepsi" in sys.argv:          # digerleri sunucu.py uzerinden aninda gelir
                out.append(region_payload(key, rows, 0.25, x, y, k_default))
    return out


def hero_grid():
    """Ana sayfa haritasi ve kuresi: 1 derecelik hucrelerde VIIRS tespiti. Dunya veri seti varsa ondan, yoksa bolgelerden.
    v: toplam; m: ay ay toplam (hucre basina 12 sayi, tam yillar); tr: son 5 tam yilin ilk 5 tam yila gore % degisimi."""
    acc, mon, yearly, rec = [], [], [], []
    src, yrs = "bölgeler", set()
    wd = HERE / "dataset" / "DUNYA"
    files = sorted(wd.glob("T*_[0-9][0-9][0-9][0-9].csv.gz")) if wd.exists() else []
    ek, ek_y = None, None                   # dunya_guncelle.py'nin ekledigi son gunler (manifestte "ek")
    if files:
        src = "dünya"
        try:
            my = json.loads((wd / "manifest.json").read_text("utf-8")).get("yillar", {})
            ek_y = max(int(y) for y in my)
            ek = my[str(ek_y)].get("ek")
            if ek and not (str(ek["bas"])[:4] == str(ek["son"])[:4] == str(ek_y)):
                ek = None
        except Exception:
            ek = None
    else:
        files = [Path(f) for f in glob.glob(str(HERE / "dataset" / "*_[0-9][0-9][0-9][0-9].csv.gz"))]
    # ince izgara (0,25 derece, Hesapla haritasiyla ayni hucre): kure ve duz harita bunu cizer
    HW, HN = 1440, 1440 * 720
    yrs_pre = sorted({int(f.name[-11:-7]) for f in files})
    full_pre = set(yrs_pre[:-1] if len(yrs_pre) >= 2 else yrs_pre)
    hv = np.zeros(HN, np.float64)
    hm = np.zeros(12 * HN, np.float32)
    hr = np.zeros(HN, np.float32)

    def _add(arr, idx, w):
        u, inv = np.unique(idx, return_inverse=True)
        arr[u] += np.bincount(inv, weights=np.asarray(w, float))
    for f in files:
        try:
            d = pd.read_csv(f, usecols=lambda c: c in ("gun", "lat_c", "lon_c", "n_VIIRS_SNPP", "n_VIIRS_NOAA20", "n_VIIRS_NOAA21", "n_Terra", "n_Aqua", "n_VIIRS_ort"))
        except Exception:
            continue
        if d.empty:
            continue
        yr = int(f.name[-11:-7])
        yrs.add(yr)
        v = d.get("n_VIIRS_SNPP", 0)
        if not isinstance(v, (int, float)) and v.sum() == 0:
            v = d[[c for c in ("n_VIIRS_NOAA20", "n_VIIRS_NOAA21") if c in d]].sum(axis=1)
        if not isinstance(v, (int, float)) and v.sum() == 0:
            v = d["n_Terra"] + d["n_Aqua"]
        t = pd.DataFrame({"a": np.floor(d["lat_c"]).astype(int), "o": np.floor(d["lon_c"]).astype(int), "v": v})
        acc.append(t.groupby(["a", "o"], as_index=False)["v"].sum())
        hi = (np.clip(np.floor((d["lat_c"].to_numpy(float) + 90) * 4), 0, 719) * HW
              + np.clip(np.floor((d["lon_c"].to_numpy(float) + 180) * 4), 0, HW - 1)).astype(np.int64)
        vv = t["v"].to_numpy(float)
        _add(hv, hi, vv)
        yearly.append(acc[-1].assign(y=yr))
        if "gun" in d:
            gs = d["gun"].astype(str)
            t["mo"] = gs.str[5:7].astype(int)
            mon.append(t.groupby(["a", "o", "mo"], as_index=False)["v"].sum().assign(y=yr))
            if yr in full_pre:
                _add(hm, (t["mo"].to_numpy() - 1) * HN + hi, vv)
            if ek:                          # son gunler ve gecmis yillarin ayni gunleri
                md = gs.str[5:10]
                sel = (md >= str(ek["bas"])[5:10]) & (md <= str(ek["son"])[5:10])
                if sel.any():
                    val = t["v"]
                    if yr == ek_y:
                        val = (d["n_VIIRS_ort"].fillna(d["n_VIIRS_SNPP"]) if "n_VIIRS_ort" in d else t["v"])
                        sm = sel.to_numpy()
                        _add(hr, hi[sm], np.asarray(val, float)[sm])
                    r_ = pd.DataFrame({"a": t["a"], "o": t["o"], "v": val})[sel]
                    rec.append(r_.groupby(["a", "o"], as_index=False)["v"].sum().assign(y=yr))
    if not acc:
        return None
    g = pd.concat(acc, ignore_index=True).groupby(["a", "o"], as_index=False)["v"].sum().sort_values("v")
    # okyanustaki tek tuk tespitler (gemi, platform, gurultu) haritada kumlanma yapiyor: cok az tespitli hucreleri at
    g = g[g["v"] >= (5 if src == "dünya" else 1)].reset_index(drop=True)
    out = {"src": src, "y0": min(yrs), "y1": max(yrs), "lat": g["a"].astype(int).tolist(), "lon": g["o"].astype(int).tolist(),
           "v": g["v"].round().astype(int).tolist()}
    idx = pd.MultiIndex.from_arrays([g["a"], g["o"]])
    full = sorted(yrs)[:-1] if len(yrs) >= 2 else sorted(yrs)          # son yil eksik olabilir: aylik ve egilimde disarida
    try:
        if mon and full:
            m = pd.concat(mon, ignore_index=True)
            m = m[m["y"].isin(full)].groupby(["a", "o", "mo"])["v"].sum().unstack("mo", fill_value=0)
            m = m.reindex(columns=range(1, 13), fill_value=0).reindex(idx, fill_value=0)
            out["m"] = m.to_numpy().round().astype(int).ravel().tolist()
            out["mY"] = [full[0], full[-1]]
        if len(full) >= 6:
            n = min(5, len(full) // 2)
            e_y, l_y = full[:n], full[-n:]
            yy = pd.concat(yearly, ignore_index=True)
            ev = yy[yy["y"].isin(e_y)].groupby(["a", "o"])["v"].sum().reindex(idx, fill_value=0).to_numpy(float)
            lv = yy[yy["y"].isin(l_y)].groupby(["a", "o"])["v"].sum().reindex(idx, fill_value=0).to_numpy(float)
            ok = (ev + lv) >= 60                                           # az tespitli hucrede egilim gurultu olur
            pc = np.where(ev > 0, 100 * (lv / np.maximum(ev, 1e-9) - 1), 300)
            pc = np.clip(np.round(pc), -100, 300)
            out["tr"] = [int(x) if k else None for x, k in zip(pc, ok)]
            out["trY"] = [e_y[0], e_y[-1], l_y[0], l_y[-1]]
        if ek and rec:
            rr = pd.concat(rec, ignore_index=True)
            now = rr[rr["y"] == ek_y].groupby(["a", "o"])["v"].sum()
            past_y = sorted(set(yrs) - {ek_y})
            base = (rr[rr["y"] != ek_y].groupby(["a", "o"])["v"].sum() / max(len(past_y), 1)) if past_y else pd.Series(dtype=float)
            if len(now):
                nv = now.reindex(idx, fill_value=0)
                nd, bd = now.to_dict(), base.to_dict()
                box = lambda dct, a, o: sum(dct.get((a + i, o + j), 0.0) for i in (-1, 0, 1) for j in (-1, 0, 1))
                top = []
                for (a, o), v in now.sort_values(ascending=False).items():
                    if len(top) >= 8 or v < 20:
                        break
                    if any(abs(a - t_["la"]) <= 3 and abs(o - t_["lo"]) <= 3 for t_ in top):
                        continue                                        # ayni yangin bolgesini bir kez say
                    n3, b3 = box(nd, a, o), box(bd, a, o)
                    top.append({"la": int(a), "lo": int(o), "v": int(round(n3)), "x": round(n3 / b3, 1) if b3 >= 5 else None})
                top.sort(key=lambda t_: -t_["v"])
                d0, d1 = pd.Timestamp(ek["bas"]), pd.Timestamp(ek["son"])
                out["rc"] = {"bas": str(ek["bas"])[:10], "son": str(ek["son"])[:10], "gun": int((d1 - d0).days + 1),
                             "v": nv.round().astype(int).tolist(), "top": top, "by": [past_y[0], past_y[-1]] if past_y else None}
    except Exception as e:
        print("  [bilgi] küre için aylık/eğilim/son günler verisi hazırlanamadı:", e)
    try:
        def enc(a, thr):
            """0,25 derecelik izgara -> 1 bayt/hucre (0: bos, 1-255: log olcekli deger) -> zlib -> base64"""
            a = np.nan_to_num(np.asarray(a, np.float64))
            a = np.where(a >= thr, a, 0.0)
            mx = float(a.max()) if a.size else 0.0
            if mx <= 0:
                return None
            q = np.where(a > 0, 1 + np.round(254 * np.log1p(a) / np.log1p(mx)), 0).astype(np.uint8)
            return base64.b64encode(zlib.compress(q.tobytes(), 9)).decode("ascii")
        hi_out = {"w": HW, "h": 720, "v": enc(hv, 3 if src == "dünya" else 1)}
        if "m" in out:
            hi_out["m"] = enc(hm, 1.5)
        if "rc" in out:
            hi_out["r"] = enc(hr, 1)
        if hi_out["v"]:
            out["hi"] = hi_out
            print(f"  ince küre ızgarası (0,25°): {sum(len(x) for x in hi_out.values() if isinstance(x, str)) / 1e6:.1f} MB")
    except Exception as e:
        print("  [bilgi] 0,25 derecelik küre ızgarası hazırlanamadı:", e)
    return out


def drift_world():
    m = HERE / "dataset" / "DUNYA" / "manifest.json"
    if not m.exists():
        return None
    yy = json.loads(m.read_text("utf-8")).get("yillar", {})
    ys = sorted(int(y) for y in yy)
    if not ys:
        return None
    r = lambda y, k: round(yy[str(y)][k] / max(yy[str(y)]["n_VIIRS_SNPP"], 1), 4)
    return {"years": ys, "terra": [r(y, "n_Terra") for y in ys], "aqua": [r(y, "n_Aqua") for y in ys],
            "snpp": [yy[str(y)]["n_VIIRS_SNPP"] for y in ys]}


def drift24():
    m = HERE / "dataset" / "TR_manifest.csv"
    if not m.exists():
        return None
    t = pd.read_csv(m).sort_values("yil")
    col = lambda c: clean(t[c].round(3).tolist()) if c in t else [None] * len(t)
    rat = lambda a: clean([round(float(x) / float(s), 4) if s and s > 0 else None
                           for x, s in zip(t[a], t.get("n_VIIRS_SNPP", pd.Series([0] * len(t))))])
    return {"years": t["yil"].astype(int).tolist(), "terra": col("yerel_saat_Terra"), "aqua": col("yerel_saat_Aqua"),
            "snpp": col("yerel_saat_VIIRS_SNPP"), "noaa20": col("yerel_saat_VIIRS_NOAA20"),
            "terraRatio": rat("n_Terra"), "aquaRatio": rat("n_Aqua")}


def main():
    g = pd.read_csv(need(OUT / "grid_cell_day.csv"))
    curve = pd.read_csv(need(OUT / "transfer_curve.csv"))
    ladder = pd.read_csv(need(OUT / "uyum_merdiveni.csv"))
    pers = pd.read_csv(need(OUT / "kalicilik.csv"))
    cx, cy = comb_curve(curve)

    g["MODIS"] = g["n_Terra"] + g["n_Aqua"]
    raw = f_comb(g["n_VIIRS_ort"], cx, cy)
    days_all = sorted(g["gun"].unique())
    cut = days_all[int(len(days_all) * 0.6)]
    tr = (g["gun"] < cut).to_numpy()
    k = float(g.loc[tr, "MODIS"].sum() / max(raw[tr].sum(), 1e-9))
    g["HARM"] = k * raw
    test_ratio = float(g.loc[~tr, "HARM"].sum() / max(g.loc[~tr, "MODIS"].sum(), 1))
    empty_share = float(g.loc[g["n_VIIRS_ort"] == 0, "MODIS"].sum() / max(g["MODIS"].sum(), 1))
    daily = g.groupby("gun")[["n_Terra", "n_Aqua", "MODIS", "n_VIIRS_ort", "HARM"]].sum().sort_index()
    r_daily = float(np.corrcoef(np.log1p(daily["MODIS"]), np.log1p(daily["HARM"]))[0, 1])

    eslesme = ladder[ladder["tur"] == "eslesme"]
    korel = ladder[ladder["tur"] == "korelasyon"]
    head_ = eslesme[eslesme["MODIS->VIIRS_%"] >= 75]
    headline = head_.iloc[0] if len(head_) else eslesme.iloc[-1]

    print("Bölgeler hazırlanıyor:")
    regions = dataset_regions(cx, cy, k)
    if not any(r["key"] == REGION for r in regions):          # cok yillik veri yoksa son 30 gunle calis
        rows = pd.DataFrame({"lat": (g["ci"] + 0.5) * CELL, "lon": (g["cj"] + 0.5) * CELL,
                             "d": (pd.to_datetime(g["gun"]) - DAY0).dt.days, "M": g["MODIS"], "V": g["n_VIIRS_ort"]})
        regions.insert(0, region_payload(REGION, rows, CELL, cx, cy, k,
                                         note="Şimdilik sadece son 30 gün. Çok yıllı takvim için build_dataset.py çalıştırılıp uygulama yeniden üretilecek."))
        regions[0]["name"] = REGION_NAME.get(REGION, REGION) + " · son 30 gün"

    default_curve = {"x": cx.round(4).tolist(), "y": cy.round(5).tolist()}
    k_up = k
    for r_ in regions:
        if r_["key"] == REGION and r_.get("curve"):
            default_curve, k_up = r_["curve"], r_["k"]
    for r_ in regions:
        r_.pop("curve", None); r_.pop("k", None)
    data = {
        "region": REGION, "regionName": REGION_NAME.get(REGION, REGION),
        "days": list(daily.index), "terra": daily["n_Terra"].round(2).tolist(), "aqua": daily["n_Aqua"].round(2).tolist(),
        "modis": daily["MODIS"].round(2).tolist(), "viirs": daily["n_VIIRS_ort"].round(2).tolist(),
        "harm": daily["HARM"].round(2).tolist(), "rDaily": round(r_daily, 2), "ratioTotal": round(test_ratio, 2),
        "k": round(k, 2), "kDefault": round(k_up, 4), "cut": str(cut), "emptyShare": round(100 * empty_share, 0),
        "curve": default_curve,
        "regions": regions,
        "ladder": {"label": [pretty_level(x) for x in eslesme["seviye"]],
                   "mv": eslesme["MODIS->VIIRS_%"].round(1).tolist(),
                   "vm": [None if pd.isna(x) else round(float(x), 1) for x in eslesme["VIIRS->MODIS_%"]]},
        "corr": {"label": [pretty_corr(x) for x in korel["seviye"]], "r": korel["Pearson_log"].round(2).tolist()},
        "headline": {"label": pretty_level(headline["seviye"]), "value": round(float(headline["MODIS->VIIRS_%"]), 1)},
        "pers": {"label": [str(x).replace("gecis", "geçiş") for x in pers["kalicilik"]], "rate": pers["MODIS_yakaladi_pct"].round(1).tolist(),
                 "n": pers["olay"].astype(int).tolist(), "frp": pers["ort_FRP_MW"].round(0).tolist()},
        "bbox": BBOX.get(REGION, ""),
        "totals": {"modis": int(g["MODIS"].sum()), "viirs": int(round(g["n_VIIRS_ort"].sum()))},
    }
    data["evidence"] = sorted(EVIDENCE, key=lambda e: ({"yüksek": 0, "orta": 1, "düşük": 2}[e["guven"]], -(e["isabet"] or 0) if e["key"] != "TR" else -999))
    data["hero"] = hero_grid()
    data["drift24"] = drift24()
    data["driftWorld"] = drift_world()
    last = None
    for r_ in regions:
        if r_["key"] == "TR":
            last = None
    trf = sorted((HERE / "dataset").glob("TR_[0-9][0-9][0-9][0-9].csv.gz"))
    if trf:
        last = str(pd.read_csv(trf[-1], usecols=["gun"])["gun"].max())
    try:                                    # dunya_guncelle.py son gunleri eklediyse (manifestte "ek") en son gun odur
        my = json.loads((HERE / "dataset" / "DUNYA" / "manifest.json").read_text("utf-8")).get("yillar", {})
        ek = max((v.get("ek") or {}).get("son", "") for v in my.values()) if my else ""
        if ek and (last is None or ek[:10] > last[:10]):
            last = ek[:10]
    except Exception:
        pass
    data["lastDay"] = last
    short = pers[pers["kalicilik"].isin(["1 gecis", "2 gecis"])]
    data["shortShare"] = round(100 * short["olay"].sum() / pers["olay"].sum(), 0)

    dh, dr = HERE / "drift_history.csv", HERE / "drift_ratios.csv"
    if dh.exists() and dr.exists():
        h = pd.read_csv(dh).pivot(index="yil", columns="uydu", values="saat")
        r = pd.read_csv(dr).set_index("yil")
        data["drift"] = {"years": [int(y) for y in h.index],
                         "aqua": clean(h.get("Aqua D", pd.Series(dtype=float)).round(3).tolist()),
                         "terra": clean(h.get("Terra D", pd.Series(dtype=float)).round(3).tolist()),
                         "snpp": clean(h.get("SNPP D", pd.Series(dtype=float)).round(3).tolist()),
                         "terraRatio": clean(r["oran_Terra/SNPP_sayi"].round(3).tolist()),
                         "aquaRatio": clean(r["oran_Aqua/SNPP_sayi"].round(3).tolist())}

    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__REGION__", data["regionName"])
    out_dir = HERE / "app"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "ates_takvimi.html"
    doc = ('<!doctype html>\n<html lang="tr">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           '<style>:root{color-scheme:light}body{margin:0}</style>\n</head>\n<body>\n' + html + '\n</body>\n</html>\n')
    path.write_text(doc, encoding="utf-8")
    if "--fragment" in sys.argv:
        (out_dir / "ates_takvimi_fragment.html").write_text(html, encoding="utf-8")
    print(f"Yazildi: {path}  ({len(html) / 1e6:.1f} MB)")
    print(f"30 günlük kalibrasyon k={k:.2f}, test isabeti %{100 * test_ratio:.0f}, günlük seyir r={r_daily:.2f}")


TEMPLATE = r"""<title>Ateş Takvimi</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="/yerel/fonts.css">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;800&family=Public+Sans:ital,wght@0,400;0,600;0,700;1,400&display=swap">
<style>
/* Renk: termal kamera skalasi (acik temada sicak = koyu, koyu temada sicak = parlak), sensor basina tek renk.
   Yazi: Big Shoulders Display yalnizca buyuk basliklar; geri kalan her sey (metin, etiket, sayi) Public Sans. */
:root{
  --ground:#eceee8; --ground2:#e3e6df; --ground3:#d9ddd4; --ink:#181c21; --ink2:#575e66; --line:#c4c8c0; --cell0:#dcdfd7;
  --viirs:#2d5f8b; --modis:#9a5b16; --harm:#c0392b; --peak:rgba(192,57,43,.13);
  --terra:#9a5b16; --aqua:#1f7a6d; --snpp:#2d5f8b;
  --t0:#f3e6b4; --t1:#efa752; --t2:#d95a2b; --t3:#a3223f; --t4:#4c1456;
  --f-disp:"Big Shoulders Display","Arial Narrow","Roboto Condensed",sans-serif;
  --f-body:"Public Sans",system-ui,"Segoe UI",Arial,sans-serif;
  --f-ui:"Public Sans",system-ui,"Segoe UI",Arial,sans-serif;
  --f-mono:var(--f-ui);
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --ground:#0e1320; --ground2:#141a2a; --ground3:#1b2336; --ink:#e3e6ea; --ink2:#9aa1ad; --line:#2a3245; --cell0:#192032;
  --viirs:#62acd9; --modis:#dc9d42; --harm:#ff6a4b; --peak:rgba(255,106,75,.15);
  --terra:#dc9d42; --aqua:#5fc2b5; --snpp:#62acd9;
  --t0:#2a1a44; --t1:#6d1c62; --t2:#bb3443; --t3:#ef7c25; --t4:#fbe08a; color-scheme:dark}}
:root[data-theme="dark"]{
  --ground:#0e1320; --ground2:#141a2a; --ground3:#1b2336; --ink:#e3e6ea; --ink2:#9aa1ad; --line:#2a3245; --cell0:#192032;
  --viirs:#62acd9; --modis:#dc9d42; --harm:#ff6a4b; --peak:rgba(255,106,75,.15);
  --terra:#dc9d42; --aqua:#5fc2b5; --snpp:#62acd9;
  --t0:#2a1a44; --t1:#6d1c62; --t2:#bb3443; --t3:#ef7c25; --t4:#fbe08a; color-scheme:dark}

*{box-sizing:border-box}
:root *{font-variant-numeric:tabular-nums!important}   /* sayilar alt alta hizali (tek aralikli yazi yerine) */
body{margin:0; background:var(--ground); color:var(--ink); font:17px/1.55 var(--f-body)}
[hidden]{display:none!important}
#exmap{-webkit-user-select:none;user-select:none}
a{color:var(--ink); text-underline-offset:3px}
a:focus-visible, button:focus-visible, select:focus-visible, input:focus-visible{outline:2px solid var(--harm); outline-offset:2px}
.wrap{width:100%; max-width:1180px; margin:0 auto; padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px))}
.n{font:600 .95em var(--f-ui); color:var(--harm); white-space:nowrap}

/* --- ust cubuk --- */
.bar{position:sticky; top:0; z-index:30; background:var(--ground); border-bottom:1.5px solid var(--ink)}
.bar .wrap{display:flex; align-items:center; gap:28px; min-height:56px}
.brand{display:flex; align-items:center; gap:10px; color:var(--ink); text-decoration:none; font:800 24px/1 var(--f-disp); text-transform:uppercase; letter-spacing:.01em; white-space:nowrap}
.brand svg{width:26px; height:26px; flex:none}
.nav{display:flex; gap:22px; overflow-x:auto; scrollbar-width:none; flex:1 1 auto; min-width:0; font:600 13.5px var(--f-ui)}
.nav::-webkit-scrollbar{display:none}
.nav a{color:var(--ink2); text-decoration:none; padding:6px 0 4px; border-bottom:2px solid transparent; white-space:nowrap}
.nav a:hover{color:var(--ink)}
.nav a[aria-current="page"]{color:var(--ink); border-bottom-color:var(--harm)}
.status{margin:0; font:400 13px var(--f-ui); color:var(--ink2); white-space:nowrap}
.langbtn{flex:none; font:600 13.5px var(--f-ui); background:none; color:var(--ink); border:1.5px solid var(--ink); padding:5px 9px; cursor:pointer}
.langbtn:hover{background:var(--ink); color:var(--ground)}
@media (max-width:860px){ .status{display:none} .bar .wrap{gap:14px} .brand span{display:none} .nav{gap:16px} }

/* --- sayfa iskeleti --- */
main{display:block}
.pg{padding-block:36px 72px; display:flex; flex-direction:column; gap:48px}
.pg[data-page="ana"]{padding-top:0}
.head h1{font:800 clamp(48px,8vw,96px)/.88 var(--f-disp); margin:0; letter-spacing:.005em}
.head p{margin:12px 0 0; font-size:18px; color:var(--ink2); max-width:60ch}
h2{font:600 34px/1 var(--f-disp); margin:0; text-wrap:balance}
h3{font:600 22px/1.05 var(--f-disp); margin:0}
p{max-width:68ch}

/* --- ana sayfa --- */
.hero{position:relative; background:var(--ground2); border-bottom:1.5px solid var(--ink)}
.hero-map{max-width:1440px; margin:0 auto; position:relative}
.hero-map svg{width:100%; height:auto; overflow:hidden}
@media (max-width:640px){ .hero-key{position:static!important; margin:8px 16px 0} }
.hero-key{position:absolute; left:max(16px, env(safe-area-inset-left, 0px)); bottom:12px; right:16px; font:400 13px/1.6 var(--f-ui); color:var(--ink2);
  display:flex; align-items:center; gap:10px; background:color-mix(in srgb, var(--ground2) 86%, transparent); padding:5px 8px; width:fit-content; max-width:calc(100% - 32px)}
.hero-key i{display:inline-block; width:96px; height:8px; flex:none}
#globe{display:block; width:100%; height:480px; cursor:grab; touch-action:pan-y; outline-offset:-3px}
#globe:active{cursor:grabbing}
#globe:focus-visible{outline:2px solid var(--harm)}
.hero.space{--ground:#0e1320; --ground2:#141a2a; --ground3:#1b2336; --ink:#e3e6ea; --ink2:#9aa1ad; --line:#2a3245; --harm:#ff6a4b; --viirs:#62acd9;
  --terra:#dc9d42; --aqua:#5fc2b5; --snpp:#62acd9; --t0:#2a1a44; --t1:#6d1c62; --t2:#bb3443; --t3:#ef7c25; --t4:#fbe08a; background:#080b14; color:var(--ink); color-scheme:dark}
.hero.space .hero-key{position:static; margin:0 16px; padding:8px 0 12px; background:none; max-width:none}
.hero-tools{position:absolute; top:12px; right:16px; display:flex; flex-direction:column; align-items:flex-end; gap:8px}
.hrow{display:flex; flex-wrap:wrap; justify-content:flex-end; align-items:center; gap:8px}
.gclock{font:600 13.5px var(--f-ui); color:var(--ink2); font-variant-numeric:tabular-nums}
.hero-tools .seg{background:var(--ground2)}
.gbar{display:flex; align-items:center; gap:10px; background:var(--ground2); border:1.5px solid var(--ink); padding:6px 10px; font:600 13.5px var(--f-ui)}
.gbar .btn{padding:4px 10px; font-size:12px}
.gbar input{width:150px; accent-color:var(--harm)}
.gbar b{display:inline-block; min-width:9ch; font:800 20px/1 var(--f-disp)}
#gyear b{min-width:4ch} #gyear span{color:var(--ink2)}
.tourgo{position:absolute; top:12px; left:max(16px, env(safe-area-inset-left, 0px)); padding:7px 12px; font-size:12px}
.tour{position:absolute; top:12px; left:max(16px, env(safe-area-inset-left, 0px)); width:min(340px, calc(100% - 32px)); background:var(--ground2); border:1.5px solid var(--ink); padding:14px 16px 16px; z-index:2}
.tour-n{margin:0 0 6px; font:600 13px var(--f-ui); color:var(--ink2)}
.tour h3{margin:0 0 8px; font:800 28px/1 var(--f-disp)}
.tour p#tourp{margin:0 0 14px; font-size:15px; line-height:1.45}
.tour-b{display:flex; flex-wrap:wrap; align-items:center; gap:8px}
.tour-b .btn{padding:6px 12px; font-size:12px}
.tour-b .btn:disabled{opacity:.4; cursor:default}
.tour-b .mini{all:unset; cursor:pointer; margin-left:auto; font:400 13px var(--f-ui); color:var(--ink2); text-decoration:underline; text-underline-offset:3px}
.tour i{position:absolute; left:0; bottom:0; height:3px; width:0; background:var(--harm)}
.gpick{position:absolute; top:56px; left:max(16px, env(safe-area-inset-left, 0px)); display:flex; flex-direction:column; align-items:flex-start; gap:8px; font:600 13.5px var(--f-ui); max-width:260px}
.gpick span:first-child{background:var(--ground2); padding:3px 6px}
#gpickmsg{color:var(--harm); background:var(--ground2)}
@media (max-width:640px){ .hero-tools, .gpick, .tour, .tourgo{position:static; margin:8px 16px 0; align-items:flex-start} .hrow{justify-content:flex-start} }
.intro{display:grid; grid-template-columns:minmax(0,7fr) minmax(0,5fr); gap:36px 56px; align-items:start}
.intro h1{font:800 clamp(64px,11vw,140px)/.84 var(--f-disp); text-transform:uppercase; letter-spacing:.005em; margin:0 0 22px}
.lead{font-size:clamp(19px,2.1vw,22px); line-height:1.5; margin:0; max-width:46ch}
.cta{display:flex; flex-wrap:wrap; gap:12px; margin-top:26px}
.btn{display:inline-flex; align-items:center; font:600 14.5px var(--f-ui); text-decoration:none; padding:10px 16px; cursor:pointer; border:1.5px solid var(--ink); color:var(--ink); background:var(--ground)}
.btn.primary{background:var(--ink); color:var(--ground)}
.btn:hover{background:var(--ink); color:var(--ground)}
.btn.primary:hover{background:var(--harm); border-color:var(--harm); color:var(--ground)}
.passlog{border:1.5px solid var(--ink)}
.passlog h2{font:600 26px/1 var(--f-disp); padding:14px 14px 8px; margin:0}
.passlog p{font:400 13px/1.6 var(--f-ui); color:var(--ink2); margin:0; padding:0 14px 12px}
table.pass{width:100%; border-collapse:collapse; font:400 13.5px var(--f-ui); font-variant-numeric:tabular-nums}
table.pass th{white-space:nowrap; font-weight:600; color:var(--ink2); text-align:left; padding:7px 14px; border-top:1.5px solid var(--ink)}
table.pass td{padding:9px 14px; border-top:1px solid var(--line); vertical-align:top}
table.pass td.sat{font:600 20px/1.1 var(--f-disp)}
table.pass td:not(.sat){white-space:nowrap}
table.pass td.sat i{display:inline-block; width:10px; height:10px; margin-right:8px}
table.pass .moved{color:var(--harm); font-weight:600}
table.pass small{display:block; color:var(--ink2); font:400 12.5px/1.6 var(--f-ui); margin-top:3px}
.routes{border-top:1.5px solid var(--ink)}
.routes a{display:grid; grid-template-columns:minmax(0,4fr) minmax(0,7fr); gap:6px 40px; padding:20px 0; border-bottom:1px solid var(--line);
  text-decoration:none; color:var(--ink); align-items:baseline}
.routes a b{font:800 clamp(32px,4.4vw,52px)/.9 var(--f-disp)}
.routes a span{color:var(--ink2); font-size:16px; max-width:56ch}
.routes a:hover b{color:var(--harm)}
.proof{font-size:clamp(19px,2.1vw,22px); line-height:1.5; max-width:60ch; margin:0}
/* --- yorunge kaymasi karti (ana sayfa) --- */
.shift{padding-bottom:22px; border-bottom:1px solid var(--line)}
.shift-grid{display:grid; grid-template-columns:1fr; gap:22px}
.shift-clock svg{display:block; width:100%; height:auto; overflow:visible}
.shift-clock svg text{font:400 13px var(--f-ui); fill:var(--ink2)}
.shift-clock svg .sat{font:600 17px var(--f-disp); fill:var(--ink)}
.shift-cap{font:400 13px/1.6 var(--f-ui); color:var(--ink2); margin:10px 0 0; max-width:70ch}
.shift-nums{margin:0; display:grid; grid-template-columns:minmax(0,1.45fr) minmax(0,1fr) minmax(0,1fr); gap:0 32px}
.shift-nums div{padding:12px 0 0; border-top:1px solid var(--line)}
.shift-nums p{grid-column:1/-1}
.shift-nums dt{font:600 13.5px var(--f-ui); color:var(--ink2); margin:0 0 4px}
.shift-nums dd{margin:0}
.shift-nums dd b{display:block; font:800 clamp(30px,3.2vw,42px)/.95 var(--f-disp); white-space:nowrap; font-variant-numeric:tabular-nums; color:var(--ink)}
.shift-nums dd b.hot{color:var(--harm)}
.shift-nums dd span{display:block; font-size:14px; color:var(--ink2); margin-top:6px; max-width:34ch}
.shift-nums p{margin:14px 0 0; font-size:16px; max-width:40ch}
@media (max-width:760px){ .shift-nums{grid-template-columns:1fr} .shift-nums div{padding-bottom:12px} }
@media (max-width:860px){ .intro, .routes a{grid-template-columns:1fr} }

/* --- yontem --- */
.sec{display:grid; grid-template-columns:minmax(0,1fr) minmax(0,3fr); gap:20px 44px; padding-top:36px; border-top:1.5px solid var(--ink)}
.side{display:flex; flex-direction:column; gap:12px; min-width:0}
.side p{margin:0; font-size:15px; color:var(--ink2); max-width:40ch}
.side .how{font:400 13px/1.6 var(--f-ui); color:var(--ink2)}
.plot{min-width:0; display:flex; flex-direction:column; gap:14px}
.scroll{overflow-x:auto; -webkit-overflow-scrolling:touch}
svg{display:block; overflow:visible}
svg text{font-family:var(--f-ui); font-size:12px; fill:var(--ink2)}
svg .lab{fill:var(--ink)}
svg .big{font-family:var(--f-disp); font-weight:800; fill:var(--ink)}
svg .grid{stroke:var(--line); stroke-width:1}
svg [data-tip]{cursor:crosshair}
.scene{background:var(--ground2)}
.cap{font:400 13px/1.7 var(--f-ui); color:var(--ink2); margin:0}
code{font:400 13.5px var(--f-ui); background:var(--ground2); padding:1px 5px}
table.ev{border-collapse:collapse; width:100%; font:400 13.5px var(--f-ui); font-variant-numeric:tabular-nums}
table.ev th{font-weight:600; color:var(--ink2); text-align:left; padding:6px 10px 6px 0; border-bottom:1.5px solid var(--ink)}
table.ev td{padding:7px 10px 7px 0; border-bottom:1px solid var(--line); white-space:nowrap}
table.ev td:first-child{white-space:normal; min-width:150px; font:400 15px var(--f-body)}
table.ev td small{display:block; color:var(--ink2); font-size:10.5px}
table.ev .r{text-align:right}
.g{display:inline-block; padding:1px 7px; font:600 13px var(--f-ui); border:1px solid currentColor}
.g-yüksek{color:var(--harm)}
.g-orta{color:var(--modis)}
.g-düşük{color:var(--ink2)}
.now{display:grid; grid-template-columns:minmax(0,4fr) minmax(0,8fr); gap:20px 56px; align-items:start; border-top:1.5px solid var(--ink); padding-top:18px}
.now h2{font:800 clamp(32px,4.4vw,52px)/.9 var(--f-disp); margin:0 0 12px}
.now p{margin:0; color:var(--ink2); font-size:15px; max-width:44ch}
.nowtab td.sat button.nowgo{all:unset; cursor:pointer; text-decoration:underline; text-underline-offset:3px; text-decoration-thickness:1px}
.nowtab td.sat button.nowgo:hover{color:var(--harm)}
.nowtab .mini{all:unset; cursor:pointer; font:400 13px var(--f-ui); text-decoration:underline; text-underline-offset:3px; color:var(--ink2); white-space:nowrap}
.nowtab .mini:hover{color:var(--harm)}
.nowtab td.sat button.nowgo:focus-visible, .nowtab .mini:focus-visible{outline:2px solid var(--harm); outline-offset:2px}
@media (max-width:860px){ .now{grid-template-columns:1fr} }
.exports{display:flex; flex-wrap:wrap; gap:8px; margin-left:auto}
details.wsets > summary, details.adv > summary{cursor:pointer; font:400 13px var(--f-ui); color:var(--ink2); width:fit-content}
details.wsets > summary:hover, details.adv > summary:hover{color:var(--ink)}
details.wsets[open] > summary{margin-bottom:6px}
details.adv{align-self:center}
details.adv[open]{flex-basis:100%; order:9}
.advbox{display:flex; flex-wrap:wrap; gap:14px 22px; align-items:end; margin-top:10px}
.controls button.act.repbtn{display:inline-flex; align-items:center; gap:8px; background:var(--ink); color:var(--ground)}
.controls button.act.repbtn:hover{background:var(--harm); border-color:var(--harm); color:var(--ground)}
.controls button.repbtn svg{width:15px; height:15px; flex:none}
.filebtn{cursor:pointer}
.filebtn input{display:block; margin-top:4px; font:400 13px var(--f-ui); max-width:230px}
.filebtn small{display:block; color:var(--ink2)}
.rep-head, .rep-foot{display:none}
body.printing .tool{max-width:780px}
@media print {
  @page{size:A4; margin:13mm}
  *{-webkit-print-color-adjust:exact; print-color-adjust:exact}
  body{background:#fff}
  .bar, .foot, .tool-head, #panel-region, #panel-world, #panel-upload, #exwait, #exmapcap, #tip, .pg:not([data-page="hesapla"]){display:none!important}
  .pg{padding:0; gap:0} .wrap{max-width:none; padding:0}
  .rep-head{display:block; border-bottom:1.5px solid #000; padding-bottom:8px; margin-bottom:12px}
  .rep-head h1{font:800 34px/1 var(--f-disp); margin:0 0 6px}
  .rep-head p{margin:0; font:400 12px/1.5 var(--f-ui)}
  .rep-foot{display:block; font:400 11px/1.55 var(--f-ui); border-top:1px solid #000; margin:14px 0 0; padding-top:6px}
  .results > div, .res-row > div, .work{break-inside:avoid}
  svg{max-width:100%; height:auto}
  .scroll{overflow:visible}
}
.gbox{display:block; margin-bottom:8px; color:var(--ink)}
.gbox .g{margin-right:6px}
#tip{position:fixed; z-index:40; pointer-events:none; background:var(--ink); color:var(--ground); font:400 13px/1.5 var(--f-ui);
  padding:7px 9px; max-width:260px; transform:translate(12px,12px)}
#tip b{display:block; font:800 20px/1.1 var(--f-disp)}
@media (max-width:860px){ .sec{grid-template-columns:1fr} }

/* --- hesapla (kesif araci) --- */
.tool{display:flex; flex-direction:column; gap:22px}
.tool-head{display:flex; flex-wrap:wrap; align-items:end; justify-content:space-between; gap:16px 32px}
.modes{display:flex; gap:22px; font:600 14.5px var(--f-ui)}
.modes button{all:unset; cursor:pointer; color:var(--ink2); padding-bottom:5px; border-bottom:2px solid transparent}
.modes button[aria-pressed="true"]{color:var(--ink); border-bottom-color:var(--harm)}
.modes button:focus-visible{outline:2px solid var(--harm); outline-offset:3px}
.controls{display:flex; flex-wrap:wrap; gap:14px 22px; align-items:end; font:400 13px var(--f-ui); color:var(--ink2)}
.controls label{display:flex; flex-direction:column; gap:6px}
.controls select, .controls button.act, .wsearch button{font:600 14.5px var(--f-ui); color:var(--ink); background:var(--ground); border:1.5px solid var(--ink); padding:7px 10px; border-radius:0}
.controls button.act, .wsearch button{cursor:pointer}
.controls button.act:hover, .wsearch button:hover{background:var(--ink); color:var(--ground)}
.work{display:grid; grid-template-columns:minmax(0,3fr) minmax(0,2fr); gap:24px 36px; align-items:start}
.selinfo{display:flex; flex-direction:column; gap:14px; min-width:0}
.selinfo .summary{font-size:18px; line-height:1.5; margin:0}
.drop{border:1.5px dashed var(--ink2); padding:22px; display:flex; flex-direction:column; gap:10px; font-size:15px; background:var(--ground2)}
.drop.over{border-color:var(--harm); background:var(--ground)}
.drop strong{font:800 24px/1 var(--f-disp)}
.drop input{font:400 13.5px var(--f-ui); color:var(--ink2); max-width:100%}
.msg{font:400 13.5px/1.6 var(--f-ui); color:var(--ink2); margin:0}
.msg.err{color:var(--harm)}
table.seasons{border-collapse:collapse; font:400 13.5px var(--f-ui); width:100%; font-variant-numeric:tabular-nums}
table.seasons th{font-weight:600; text-align:left; color:var(--ink2); padding:6px 10px 6px 0; border-bottom:1.5px solid var(--ink)}
table.seasons td{padding:5px 10px 5px 0; border-bottom:1px solid var(--line); white-space:nowrap}
table.seasons td.r, table.seasons th.r{text-align:right}
.results{display:flex; flex-direction:column; gap:30px}
.res-row{display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:30px}
.res-row > div{min-width:0; display:flex; flex-direction:column; gap:10px}
@media (max-width:860px){ .work{grid-template-columns:1fr} }
.world{display:flex; flex-direction:column; gap:14px}
.woff{background:var(--ground2); padding:14px 16px; font-size:15px; display:flex; flex-direction:column; gap:6px; max-width:70ch}
.wsearch{display:flex; gap:10px; flex-wrap:wrap}
.wsearch input{flex:1 1 260px; min-width:0; font:400 16px var(--f-body); color:var(--ink); background:var(--ground); border:1.5px solid var(--ink); padding:9px 12px; border-radius:0}
.wres{list-style:none; margin:0; padding:0; display:flex; flex-direction:column; max-height:236px; overflow:auto}
.wres:empty{display:none}
.wres li{border-bottom:1px solid var(--line)}
.wres button{all:unset; box-sizing:border-box; width:100%; cursor:pointer; padding:7px 6px; display:grid; grid-template-columns:minmax(0,1fr) auto; gap:1px 16px}
.wres button:hover, .wres button[aria-current="true"]{background:var(--ground2)}
.wres button:focus-visible{outline:2px solid var(--harm); outline-offset:-2px}
.wres b{font-weight:700; font-size:15px}
.wres small{grid-column:1; font:400 13px/1.5 var(--f-ui); color:var(--ink2); overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
.wres em{font:400 13px var(--f-ui); color:var(--ink2); font-style:normal; grid-row:1/3; grid-column:2; align-self:center}
.wsets{display:flex; flex-direction:column; gap:6px}
.wsets > .lab{font:400 13px var(--f-ui); color:var(--ink2)}
.chipbox{display:flex; flex-wrap:wrap; gap:6px}
.lab .mini{all:unset; cursor:pointer; margin-left:10px; text-decoration:underline; text-underline-offset:3px; color:var(--ink2)}
.lab .mini:hover{color:var(--harm)}
.lab .mini:focus-visible{outline:2px solid var(--harm); outline-offset:2px}
.chipbox button{all:unset; cursor:pointer; font:600 13.5px var(--f-ui); padding:5px 9px; border:1px solid var(--line); color:var(--ink)}
#whchips button{border-style:dashed; color:var(--ink2)}
.chipbox button:hover, .chipbox button[aria-current="true"]{border-color:var(--harm); color:var(--harm)}
.chipbox button:focus-visible{outline:2px solid var(--harm); outline-offset:2px}
.seg{display:inline-flex; border:1.5px solid var(--ink)}
.seg button{all:unset; cursor:pointer; font:600 13.5px var(--f-ui); padding:7px 10px; color:var(--ink2)}
.seg button[aria-pressed="true"]{background:var(--ink); color:var(--ground)}
.seg button:focus-visible{outline:2px solid var(--harm); outline-offset:2px}
.controls .chk{flex-direction:row; align-items:center; gap:8px}
.controls button.go{border-color:var(--harm); color:var(--harm)}
.controls button.go:hover{background:var(--harm); color:var(--ground)}
.controls button.go:disabled{opacity:.4; cursor:not-allowed; background:var(--ground); color:var(--harm)}
.wmap{height:min(460px,62vh); background:var(--ground2); border:1.5px solid var(--ink); position:relative}
.wmap.boxing{cursor:crosshair}
.wmap .leaflet-tile-pane{filter:saturate(.35) contrast(.95)}
.wmap.dark .leaflet-tile-pane{filter:invert(.92) hue-rotate(180deg) saturate(.3) brightness(.9)}
.wsel{font-size:17px; margin:0}
.wsel b{font:800 26px/1 var(--f-disp); display:block; margin-bottom:2px}
.wprog{height:6px; background:var(--ground2); border:1px solid var(--line)}
.wprog i{display:block; height:100%; width:0; background:var(--harm)}
@media (prefers-reduced-motion:no-preference){ .wprog i{transition:width .4s} }
.leaflet-container{font:400 13px var(--f-ui)!important; background:var(--ground2)!important}
.leaflet-control-attribution{font:400 12px var(--f-ui)!important}

/* --- karsilastir --- */
.cmp-pick{display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:18px}
.slot{border:1.5px solid var(--ink); border-top:7px solid var(--c); padding:14px; display:flex; flex-direction:column; gap:12px}
.slot h3{font:800 34px/1 var(--f-disp); color:var(--c)}
.slot .controls select{max-width:100%}
.picker{border:1.5px solid var(--ink); padding:16px; display:flex; flex-direction:column; gap:14px; background:var(--ground)}
.cmp-sum{font-size:19px; line-height:1.5; margin:0; max-width:66ch}
table.cmp{border-collapse:collapse; font:400 13.5px var(--f-ui); width:100%; max-width:860px; font-variant-numeric:tabular-nums}
table.cmp th{font-weight:600; color:var(--ink2); text-align:left; padding:7px 12px 7px 0; border-bottom:1.5px solid var(--ink)}
table.cmp td{padding:7px 12px 7px 0; border-bottom:1px solid var(--line)}
table.cmp td:first-child{font:400 15px var(--f-body)}
table.cmp .r{text-align:right}

/* --- hakkinda --- */
.about{display:grid; grid-template-columns:minmax(0,1fr) minmax(0,3fr); gap:28px 44px}
.about dl{margin:0; display:grid; grid-template-columns:minmax(0,1fr); gap:18px}
.about dt{font:600 22px/1.05 var(--f-disp)}
.about dd{margin:4px 0 0; color:var(--ink2); font-size:16px; max-width:64ch}
@media (max-width:860px){ .about{grid-template-columns:1fr} }
.foot{border-top:1.5px solid var(--ink); padding-block:20px 40px; font:400 13px/1.8 var(--f-ui); color:var(--ink2)}
.foot p{margin:0}
@keyframes draw{to{stroke-dashoffset:0}}
@keyframes rise{from{opacity:0}to{opacity:1}}
@media (prefers-reduced-motion:reduce){ *{animation:none!important; transition:none!important} }
</style>

<header class="bar">
  <div class="wrap">
    <a class="brand" href="#/" aria-label="Ateş Takvimi ana sayfa">
      <svg viewBox="0 0 26 26" aria-hidden="true"><ellipse cx="13" cy="13" rx="11.5" ry="5" fill="none" stroke="var(--ink)" stroke-width="1.6" transform="rotate(-28 13 13)"/><circle cx="13" cy="13" r="4.2" fill="var(--harm)"/><circle cx="22.6" cy="8" r="1.9" fill="var(--ink)"/></svg>
      <span>Ateş Takvimi</span></a>
    <nav class="nav" aria-label="Sayfalar">
      <a href="#/" data-p="ana">Ana sayfa</a>
      <a href="#/hesapla" data-p="hesapla">Hesapla</a>
      <a href="#/karsilastir" data-p="karsilastir">Karşılaştır</a>
      <a href="#/yontem" data-p="yontem">Yöntem ve kanıt</a>
      <a href="#/hakkinda" data-p="hakkinda">Hakkında</a>
    </nav>
    <p class="status" id="status"></p>
    <button type="button" class="langbtn" id="langbtn">EN</button>
  </div>
</header>

<main lang="tr">
  <!-- ================= ANA SAYFA ================= -->
  <section class="pg" data-page="ana">
    <div class="hero">
      <div class="hero-map"><div id="herosvg"></div>
        <canvas id="globe" tabindex="0" aria-label="Dünya yangın küresi: uyduların gördüğü yangınlarla çizilmiş. Ok tuşlarıyla döndür, Enter ile ortadaki yeri seç." hidden></canvas>
        <div class="hero-tools">
          <div class="hrow">
          <div class="seg" role="group" aria-label="Harita görünümü"><button type="button" id="hv-kure" aria-pressed="true">Küre</button><button type="button" id="hv-duz" aria-pressed="false">Düz harita</button></div>
          <div class="seg" role="group" aria-label="Küre görünümü" id="gmodes"><button type="button" data-gm="toplam" aria-pressed="true">Toplam</button><button type="button" data-gm="ay" aria-pressed="false">Ay ay</button><button type="button" data-gm="degisim" aria-pressed="false">Değişim</button><button type="button" data-gm="son" aria-pressed="false">Son günler</button></div>
          </div>
          <div class="hrow" id="grow2">
            <span class="gclock" id="gclock"></span>
            <div class="seg" role="group" aria-label="Uydular ve dönüş"><button type="button" id="gtrk" aria-pressed="true">Uydular</button><button type="button" id="gpp" aria-pressed="false">Durdur</button></div>
          </div>
          <div class="gbar" id="gyear" hidden><span>Yörünge yılı</span><input type="range" id="gyr" min="0" max="23" step="1" value="23" aria-label="Yörünge yılı"><b id="gyrlab"></b></div>
          <div class="gbar" id="gbar" hidden><button type="button" class="btn" id="gplay">Oynat</button><input type="range" id="gmon" min="0" max="11" step="1" value="0" aria-label="Ay"><b id="gmlab">Ocak</b></div>
        </div>
        <button type="button" class="btn primary tourgo" id="tourgo">Turu başlat</button>
        <div class="tour" id="tour" hidden>
          <p class="tour-n" id="tourn"></p><h3 id="tourh"></h3><p id="tourp"></p>
          <div class="tour-b"><button type="button" class="btn" id="tourprev">Geri</button><button type="button" class="btn primary" id="tournext">İleri</button><button type="button" class="mini" id="tourend">Turu kapat</button></div>
          <i id="tourbar"></i>
        </div>
        <div class="gpick" id="gpick" hidden><span id="gpickll"></span><button type="button" class="btn primary" id="ggo">Bu yerin takvimini hesapla</button><span id="gpickmsg"></span></div>
        <div class="hero-key" id="herokey"></div></div>
    </div>
    <div class="wrap intro">
      <div>
        <h1>Ateş Takvimi</h1>
        <p class="lead">Terra ve Aqua yirmi yılı aşkın süredir Dünya'daki yangınları sayıyor. Yakıtları bitiyor ve geçiş saatleri kayıyor. Bu kaydı VIIRS'e kesintisiz devretmek için iki sensörün sayımını aynı ölçeğe çeviriyoruz. Bir yer seç, 2003'ten bugüne uyumlu yangın takvimini gör.</p>
        <div class="cta"><a class="btn primary" href="#/hesapla">Bir yer seç</a><a class="btn" href="#/yontem">Yöntemi incele</a></div>
      </div>
      <div class="passlog">
        <h2>Uyduların geçiş saati</h2>
        <p>Türkiye üzerinden gündüz geçişinin yerel güneş saati, yıllık medyan.</p>
        <table class="pass" id="passtab"></table>
      </div>
    </div>
    <div class="wrap" id="nowsec" hidden>
      <section class="now" aria-labelledby="nowh">
        <div><h2 id="nowh">Şu an nerede yanıyor</h2><p id="nowcap"></p></div>
        <div class="scroll"><table class="pass nowtab" id="nowtab"></table></div>
      </section>
    </div>
    <div class="wrap">
      <nav class="routes" aria-label="Ne yapabilirsin">
        <a href="#/hesapla"><b>Hesapla</b><span>Bir ülke, il, ilçe ya da haritada çizdiğin bir alan için uyumlu yangın takvimi, yıllık seyir, kritik aylar ve sezon tablosu.</span></a>
        <a href="#/karsilastir"><b>Karşılaştır</b><span>İki yeri ya da aynı yerin iki dönemini yan yana koy: hangisinde sezon daha erken başlıyor, hangi aylar ağırlıkta.</span></a>
        <a href="#/yontem"><b>Yöntem ve kanıt</b><span>Uyumun hangi ölçekte tuttuğu, 18 bölgedeki test sonuçları ve MODIS'in kayan saatinin kayda etkisi.</span></a>
      </nav>
    </div>
    <div class="wrap"><p class="proof" id="proof"></p></div>
  </section>

  <!-- ================= HESAPLA ================= -->
  <section class="pg" data-page="hesapla" hidden>
  <div class="wrap tool" id="tool">
    <div class="rep-head"><h1 id="reph"></h1><p id="repp"></p></div>
    <div class="tool-head">
      <div class="head"><h1>Hesapla</h1><p>Bir yer seç, MODIS ve VIIRS'in birleştirildiği yangın takvimini gör.</p></div>
      <div class="modes" role="group" aria-label="Veri kaynağı">
        <button id="mode-region" aria-pressed="true">Örnek bölgeler</button>
        <button id="mode-world" aria-pressed="false">Dünyada ara</button>
        <button id="mode-upload" aria-pressed="false">Dosya yükle</button>
      </div>
    </div>


    <div id="panel-world" class="world" hidden>
      <div class="woff" id="woff">
        <strong>Dünya araması bu bilgisayardaki sunucuyla çalışır.</strong>
        <span>Proje klasöründe <code>python sunucu.py</code> çalıştır; uygulama <code>http://localhost:8765</code> adresinde açılır. Sunucu FIRMS anahtarını saklar, verileri hazırlar ve uyumlulaştırmayı yapar. Bu sayfa sunucusuz açıldığı için örnek bölgeler ve dosya yükleme kullanılabilir.</span>
      </div>
      <div id="won" class="world" hidden>
        <form id="wform" class="wsearch" autocomplete="off">
          <input id="wq" type="search" placeholder="Ülke, il, ilçe ya da şehir yaz: Muğla, Manchester, Kaliforniya" aria-label="Yer ara">
          <button type="submit">Ara</button>
        </form>
        <ol id="wres" class="wres" aria-label="Arama sonuçları"></ol>
        <details class="wsets" id="wsets" hidden><summary class="lab"><span id="wsetlab">İndirilen bölgeler</span>, anında hesaplanır<button type="button" id="wsetre" class="mini" title="İndirmesi yeni biten bölgeleri listeye ekle">yenile</button></summary><div id="wchips" class="chipbox"></div></details>
        <details class="wsets" id="whist" hidden><summary class="lab">Son aramaların<button type="button" id="whistclr" class="mini">temizle</button></summary><div id="whchips" class="chipbox"></div></details>
        <div class="controls">
          <label>Haritaya tıklayınca seç<span class="seg" role="group" aria-label="Seçim düzeyi">
            <button type="button" data-lv="ulke" aria-pressed="false">ülke</button><button type="button" data-lv="il" aria-pressed="true">il</button><button type="button" data-lv="ilce" aria-pressed="false">ilçe</button><button type="button" data-lv="kutu" aria-pressed="false">kutu çiz</button></span></label>
          <label for="wy0">İlk yıl<select id="wy0"></select></label>
          <label for="wy1">Son yıl<select id="wy1"></select></label>
          <details class="adv"><summary>Gelişmiş</summary><div class="advbox">
            <label class="chk"><input type="checkbox" id="wall"> Tüm VIIRS uyduları</label>
            <label for="wbfile" class="filebtn">Kendi sınırını yükle<input type="file" id="wbfile" accept=".geojson,.json,.kml,application/geo+json,application/vnd.google-earth.kml+xml"><small>GeoJSON ya da KML</small></label>
          </div></details>
          <button class="act go" id="wgo" type="button" disabled>Takvimi hesapla</button>
        </div>
        <div id="wmap" class="wmap"></div>
        <p class="wsel" id="wsel">Bir yer ara ya da haritada tıkla.</p>
        <p class="msg" id="west"></p>
        <div class="wprog" id="wprog" hidden><i></i></div>
        <p class="msg" id="wmsg"></p>
      </div>
    </div>

    <div id="panel-upload" class="drop" hidden>
      <strong>FIRMS dosyalarını buraya bırak</strong>
      <span>firms.modaps.eosdis.nasa.gov/download adresinden aynı bölge ve dönem için indirdiğin MODIS ve VIIRS CSV dosyalarını seç (arşiv ya da NRT). Birden fazla dosya seçebilirsin. Hesap tamamen tarayıcında yapılır, dosyalar hiçbir yere gönderilmez.</span>
      <input type="file" id="exfile" accept=".csv,text/csv" multiple>
      <p class="msg" id="upmsg"></p>
    </div>

    <p class="woff" id="exwait" hidden><strong>Henüz bir yer seçmedin.</strong><span>Yukarıda bir yer ara, bir düğmeye bas ya da haritada tıkla; sonra Takvimi hesapla'ya bas.</span></p>
    <div id="panel-region" class="controls">
      <label for="exreg">Bölge<select id="exreg"></select></label>
      <label for="exy0">İlk yıl<select id="exy0"></select></label>
      <label for="exy1">Son yıl<select id="exy1"></select></label>
      <label for="exlayer">Haritada<select id="exlayer">
        <option value="e">uyumlulaştırılmış</option><option value="m">MODIS</option><option value="v">VIIRS ham</option></select></label>
      <button class="act" id="exclear" type="button" hidden>Seçimi temizle</button>
      <div class="exports">
        <button class="act" id="exdl" type="button">CSV indir</button>
        <button class="act" id="exshare" type="button" hidden>Bağlantıyı kopyala</button>
        <button class="act repbtn" id="exrep" type="button"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.5v8.2M4.6 6.5 8 9.9l3.4-3.4M2.5 11.5v3h11v-3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="square"/></svg>Rapor indir (PDF)</button>
      </div>
    </div>
    <div class="work">
      <div class="plot"><div id="exmap" class="scene"></div><p class="cap" id="exmapcap">Haritada sürükleyerek bir alan seç; tıklarsan o noktanın çevresindeki 1°'lik alan seçilir. Hücreler 0,25° (yaklaşık 25 km): iki sensörün uyumunun geçerli olduğu en ince ölçek.</p></div>
      <div class="selinfo">
        <p class="summary" id="exsum"></p>
        <p class="msg" id="excal"></p>
      </div>
    </div>

    <div class="results">
      <div><h3 id="excalh">Yanma takvimi</h3><div class="scroll"><div id="exgrid"></div></div></div>
      <div class="res-row">
        <div><h3>Yıllara göre</h3><div id="exyear"></div><p class="cap" id="exyearcap"></p></div>
        <div><h3>Kritik aylar</h3><div id="exmonth"></div><p class="cap">Seçilen yılların ortalaması. Turuncu sütunlar ortalamanın üstündeki aylar.</p></div>
      </div>
      <div class="res-row">
        <div><h3>Sezon kayması</h3><div id="exshift"></div><p class="cap" id="exshiftcap"></p></div>
        <div><h3>Ortalamadan sapma</h3><div id="exanom"></div><p class="cap" id="exanomcap"></p></div>
      </div>
      <div><h3>Sezon tablosu</h3><div class="scroll"><table class="seasons" id="extable"></table></div>
        <p class="cap">Sezon başı ve sonu, yıllık uyumlulaştırılmış aktivitenin %10'una ve %90'ına ulaşılan günler. Tepe, en yoğun 15 günlük pencerenin ortası.</p></div>
    </div>
    <p class="rep-foot">Veri: NASA FIRMS (MODIS C6.1, VIIRS 375 m). Yöntem: 0,25° hücre ve gün ölçeğinde VIIRS sayıları bölgeye özgü bir eğriyle MODIS eşdeğerine çevrilir; kalıcı ısı kaynakları ayıklanır. Sayılar yangın tespitidir, yanan alan değildir. Ateş Takvimi, NASA Space Apps Challenge 2026.</p>
  </div>
  </section>

  <!-- ================= KARSILASTIR ================= -->
  <section class="pg" data-page="karsilastir" hidden>
  <div class="wrap" style="display:flex;flex-direction:column;gap:32px">
    <div class="tool-head">
      <div class="head"><h1>Karşılaştır</h1><p>İki yeri ya da aynı yerin iki dönemini yan yana koy.</p></div>
      <div class="modes" role="group" aria-label="Karşılaştırma türü">
        <button id="cmp-m-yer" aria-pressed="true">İki yer</button>
        <button id="cmp-m-donem" aria-pressed="false">İki dönem</button>
      </div>
    </div>
    <div class="cmp-pick">
      <div class="slot" style="--c:var(--harm)"><h3>A</h3>
        <div class="controls"><label for="ca-r">Bölge<select id="ca-r"></select></label><label for="ca-y0">İlk yıl<select id="ca-y0"></select></label><label for="ca-y1">Son yıl<select id="ca-y1"></select></label><button class="act" type="button" id="ca-pick">Başka yer seç</button></div></div>
      <div class="slot" style="--c:var(--viirs)"><h3>B</h3>
        <div class="controls"><label for="cb-r">Bölge<select id="cb-r"></select></label><label for="cb-y0">İlk yıl<select id="cb-y0"></select></label><label for="cb-y1">Son yıl<select id="cb-y1"></select></label><button class="act" type="button" id="cb-pick">Başka yer seç</button></div></div>
    </div>
    <p class="msg" id="cmpnote"></p>
    <div id="cmp-picker" class="picker" hidden>
      <h3 id="pk-title">Yer seç</h3>
      <div class="woff" id="pk-off"><strong>Yeni yer seçmek için sunucu gerekir.</strong><span>Proje klasöründe <code>python sunucu.py</code> çalıştır. Sunucusuz açılan sayfada yalnızca örnek bölgeler karşılaştırılabilir.</span></div>
      <div id="pk-on" class="world" hidden>
        <form id="pk-form" class="wsearch" autocomplete="off">
          <input id="pk-q" type="search" placeholder="Ülke, il, ilçe ya da şehir yaz: Antalya, Attika, Kaliforniya" aria-label="Karşılaştırmak için yer ara">
          <button type="submit">Ara</button>
        </form>
        <ol id="pk-res" class="wres" aria-label="Arama sonuçları"></ol>
        <div class="wsets"><span class="lab">İndirilen bölgeler</span><div id="pk-chips" class="chipbox"></div></div>
        <div class="controls">
          <label>Haritaya tıklayınca seç<span class="seg" role="group" aria-label="Seçim düzeyi" id="pk-seg">
            <button type="button" data-plv="ulke" aria-pressed="false">ülke</button><button type="button" data-plv="il" aria-pressed="true">il</button><button type="button" data-plv="ilce" aria-pressed="false">ilçe</button></span></label>
          <button class="act go" id="pk-go" type="button" disabled>Bu yeri kullan</button>
          <button class="act" id="pk-close" type="button">Vazgeç</button>
        </div>
        <div id="pk-map" class="wmap"></div>
        <p class="wsel" id="pk-sel">Bir yer ara, bir bölgeye bas ya da haritada tıkla.</p>
        <p class="msg" id="pk-msg"></p>
      </div>
    </div>
    <p class="cmp-sum" id="cmpsum"></p>
    <div class="res-row">
      <div><h3>Ayların payı</h3><div id="cmpmon"></div><p class="cap">Her ayın yıllık uyumlulaştırılmış aktiviteden aldığı pay. Büyüklükten bağımsız, mevsim şeklini karşılaştırır.</p></div>
      <div><h3>Sezon ilerleyişi</h3><div id="cmpcum"></div><p class="cap">Yılın her gününe kadar biriken aktivitenin yüzdesi. Eğri ne kadar soldaysa sezon o kadar erken.</p></div>
    </div>
    <div><h3>Yıllara göre</h3><div id="cmpyear"></div><p class="cap">Yıllık uyumlulaştırılmış toplam (MODIS-eşdeğeri tespit).</p></div>
    <div><h3>Özet tablo</h3><div class="scroll"><table class="cmp" id="cmptab"></table></div></div>
  </div>
  </section>

  <!-- ================= YONTEM VE KANIT ================= -->
  <section class="pg" data-page="yontem" hidden>
  <div class="wrap" style="display:flex;flex-direction:column;gap:40px">
    <div class="head"><h1>Yöntem ve kanıt</h1><p>İki sensör aynı yangını farklı sayıyor. Nerede ve ne kadar birleştirilebildiklerini ölçtük.</p></div>

    <section class="sec">
      <div class="side">
        <h2>18 bölgede test</h2>
        <p>Her bölgede dönüşüm, Terra ve Aqua'nın yörüngesi henüz kaymamışken ilk yıllarda öğreniliyor ve sonraki yıllarda modelin hiç görmediği verilerle test ediliyor.</p>
        <div class="how">Güven: kalibrasyon yıllarındaki MODIS tespit sayısı ve yıllık sapmaya göre. Az yanan bölgelerde güven düşük çıkıyor; bunu gizlemiyoruz.</div>
      </div>
      <div class="plot"><div class="scroll"><table class="ev" id="evtab"></table></div></div>
    </section>

    <section class="sec">
      <div class="side">
        <h2>VIIRS'i MODIS diline çevirmek</h2>
        <p>Tek bir VIIRS uydusu (S-NPP), Terra ve Aqua'nın toplamından 4–5 kat fazla tespit yapıyor, çünkü pikseli 7 kat küçük. Ham sayıları üst üste koymak, 2012'deki sensör geçişinde takvimde yapay bir sıçrama yaratır. Bu yüzden her hücre-gündeki VIIRS sayısını MODIS'in göreceği sayıya çeviriyoruz.</p>
        <div class="how" id="calhow"></div>
      </div>
      <div class="plot"><div id="line"></div></div>
    </section>

    <section class="sec">
      <div class="side">
        <h2>Uyum hangi ölçekte tutuyor</h2>
        <p>Karşılaştırma ölçeği büyüdükçe MODIS'in tespitlerinin yüzde kaçını VIIRS de görüyor. Tek piksel düzeyinde uyum düşük; literatürde iki uydu aynı anda geçtiğinde bile tarım alanlarında bu oran %47. Takvimi uyumun eşiği geçtiği ölçekte kuruyoruz.</p>
        <div class="how">Dolu sütun: hedefi geçen ölçek. Boş sütun: geçemeyen.</div>
      </div>
      <div class="plot"><div id="steps"></div><div id="corr"></div></div>
    </section>

    <section class="sec">
      <div class="side">
        <h2>Yangın, MODIS gelmeden sönüyor</h2>
        <p id="persnote"></p>
        <div class="how">Bir olayın kaç ayrı VIIRS geçişinde görüldüğü, o yangının ne kadar sürdüğünü gösteriyor. Uzun süren yangını iki sensör de yakalıyor.</div>
      </div>
      <div class="plot"><div id="pers"></div></div>
    </section>

    <section class="sec" id="driftsec">
      <div class="side">
        <h2>MODIS'in saati kayıyor</h2>
        <p id="driftnote">Terra ve Aqua'nın yakıtı bitiyor ve yörüngeleri kayıyor. Güneş saati, uyduların Türkiye üzerinden her yıl geçtiği medyan yerel saati gösteriyor. Her halka bir yıl, içten dışa.</p>
        <div class="how">Gölgeli dilim: gün içinde yangın aktivitesinin tepe yaptığı saatler (14:00–15:00).</div>
      </div>
      <div class="plot">
        <div class="shift" id="shiftsec" hidden>
          <div class="shift-grid">
            <div class="shift-clock"><div id="shiftsvg"></div><p class="shift-cap" id="shiftcap"></p></div>
            <dl class="shift-nums" id="shiftnums"></dl>
          </div>
        </div>
        <div id="dial"></div><div id="ratio"></div>
      </div>
    </section>
    <div id="driftrule" hidden></div>
  </div>
  </section>

  <!-- ================= HAKKINDA ================= -->
  <section class="pg" data-page="hakkinda" hidden>
  <div class="wrap" style="display:flex;flex-direction:column;gap:40px">
    <div class="head"><h1>Hakkında</h1><p>NASA Space Apps Challenge 2026, Ankara. Görev: Harmonization of MODIS and VIIRS Hot Spots.</p></div>
    <div class="about">
      <h2>Neden</h2>
      <dl>
        <div><dt>Kayıt el değiştiriyor</dt><dd>MODIS 2000'den beri küresel yangın kaydının omurgası. Terra'nın veri toplaması Ocak 2027'de, Aqua'nınki 2027 sonbaharında bitiyor. Kaydın devamı VIIRS'te, ama VIIRS'in pikseli 7 kat küçük ve aynı yangını çok daha fazla sayıyor.</dd></div>
        <div><dt>Ham birleştirme yanıltır</dt><dd>İki sensörün sayılarını yan yana koymak 2012'de yapay bir sıçrama, 2021'den sonra da yörünge kaymasından gelen sahte bir eğilim üretir. Ateş Takvimi her bölge için VIIRS'i MODIS'in göreceği sayıya çevirir ve bunun ne kadar tuttuğunu test eder.</dd></div>
      </dl>
      <h2>Veri</h2>
      <dl>
        <div><dt>NASA FIRMS</dt><dd>MODIS Collection 6.1 (Terra ve Aqua, 1 km) ve VIIRS 375 m (S-NPP, NOAA-20, NOAA-21) aktif yangın tespitleri; arşiv (SP) ve yakın gerçek zamanlı (NRT) veri.</dd></div>
        <div><dt>Hazır veri setleri</dt><dd id="aboutdata"></dd></div>
        <div><dt>Harita ve arama</dt><dd>Yer arama ve sınırlar: OpenStreetMap Nominatim. Altlık harita: OpenStreetMap katkıcıları.</dd></div>
      </dl>
      <h2>Sınırlar</h2>
      <dl>
        <div><dt>Tespit sayısı yanık alan değildir</dt><dd>Takvim, uydunun gördüğü sıcak nokta sayısını gösterir; yanan hektarı değil.</dd></div>
        <div><dt>Bulut ve geçiş saati</dt><dd>Bulut altındaki yangınlar iki sensörde de görünmez. Bulut maskesi kullanılmadı.</dd></div>
        <div><dt>Ölçek</dt><dd>Birleştirme 0,25° hücre ve gün ölçeğinde geçerli. Tek tek pikselleri eşleştirmek fiziksel olarak mümkün değil: iki uydu aynı anda geçtiğinde bile tarım alanlarında uyum %47.</dd></div>
      </dl>
      <h2>Kendi bilgisayarında</h2>
      <dl>
        <div><dt>Çalıştırma</dt><dd>Proje klasöründe <code>python sunucu.py</code>. Uygulama <code>http://localhost:8765</code> adresinde açılır; dünya araması ve indirilen bölgeler bu modda çalışır.</dd></div>
        <div><dt>Kaynaklar</dt><dd>Fu ve ark. 2020, Remote Sensing 12:2870. Sofiev ve Kouznetsov 2026, GMD 19:10123. Schroeder ve ark. 2014, RSE. CAMS, "Drifting away from MODIS".</dd></div>
      </dl>
    </div>
  </div>
  </section>
</main>
<footer class="foot"><div class="wrap"><p>Veri: NASA FIRMS (MODIS C6.1, VIIRS 375 m). Harita: OpenStreetMap katkıcıları.</p><p>Ateş Takvimi, NASA Space Apps Challenge 2026 Ankara için hazırlandı.</p></div></footer>
<div id="tip" hidden></div>

<script>
const D = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const AY = ["Oca","Şub","Mar","Nis","May","Haz","Tem","Ağu","Eyl","Eki","Kas","Ara"];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const nf = (v, d = 0) => Number(v).toLocaleString("tr-TR", { minimumFractionDigits: d, maximumFractionDigits: d });
const fd = s => { const [y, m, d] = s.split("-").map(Number); return d + " " + AY[m - 1]; };
const ft = h => { if (h == null || isNaN(h)) return "–"; const x = ((h % 24) + 24) % 24; let H = Math.floor(x), M = Math.round((x - H) * 60); if (M === 60) { H = (H + 1) % 24; M = 0; } return String(H).padStart(2, "0") + ":" + String(M).padStart(2, "0"); };
function E(tag, a, p) { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; }
function T(p, x, y, txt, a) { const t = E("text", Object.assign({ x, y }, a || {}), p); t.textContent = txt; return t; }
function canvas(id, w, h) { const host = document.getElementById(id); host.innerHTML = ""; const s = E("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h, role: "img" }); host.appendChild(s); return s; }
function W(id, min) { const el = document.getElementById(id); return Math.max(min || 280, Math.floor(el.parentElement.clientWidth)); }
let STOPS = [];
function hex(c) { c = c.replace("#", ""); if (c.length === 3) c = c.split("").map(x => x + x).join(""); return [0, 2, 4].map(i => parseInt(c.substr(i, 2), 16)); }
function heat(t) { t = Math.max(0, Math.min(1, t)); const x = t * (STOPS.length - 1), i = Math.min(Math.floor(x), STOPS.length - 2), f = x - i, a = STOPS[i], b = STOPS[i + 1]; return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`; }
function tip(el, html) { el.setAttribute("data-tip", html); }
function nice(m) { const p = Math.pow(10, Math.floor(Math.log10(m))); for (const k of [1, 2, 2.5, 5, 10]) if (k * p >= m) return k * p; return 10 * p; }

/* ---------- ana sayfa ---------- */
const MONTHS = ["Ocak","Şubat","Mart","Nisan","Mayıs","Haziran","Temmuz","Ağustos","Eylül","Ekim","Kasım","Aralık"];
const fdl = s => { if (!s) return ""; const [y, m, d] = s.split("-").map(Number); return `${d} ${MONTHS[m - 1]} ${y}`; };
let HERO_DONE = false;
function groundTrack(incl, periodMin, lon0, desc) {
  /* gunduz yarisi: Terra kuzeyden guneye (inen), Aqua ve S-NPP guneyden kuzeye (cikan) */
  const i = incl * Math.PI / 180, we = 360 / 1436, out = [], u0 = desc ? Math.PI / 2 : -Math.PI / 2;
  let seg = [], prev = null;
  for (let k = 0; k <= 360; k++) {
    const u = u0 + k / 360 * Math.PI, t = k / 720 * periodMin;
    const lat = Math.asin(Math.sin(i) * Math.sin(u)) * 180 / Math.PI;
    let lon = Math.atan2(Math.cos(i) * Math.sin(u), Math.cos(u)) * 180 / Math.PI + lon0 - we * t;
    lon = ((lon + 540) % 360) - 180;
    if (Math.abs(lat) > 68) { if (seg.length) { out.push(seg); seg = []; } prev = null; continue; }
    if (prev != null && Math.abs(lon - prev) > 180) { out.push(seg); seg = []; }
    seg.push([lon, lat]); prev = lon;
  }
  out.push(seg);
  return out;
}
function hero() {
  const H = D.hero, host = document.getElementById("herosvg");
  const w = Math.max(320, Math.floor(host.parentElement.clientWidth || 1200)), LA0 = -58, LA1 = 78;
  const h = Math.round(w * (LA1 - LA0) / 360), x = lo => (lo + 180) / 360 * w, y = la => (LA1 - la) / (LA1 - LA0) * h;
  const s = canvas("herosvg", w, h);
  s.setAttribute("aria-label", "Dünya haritası: yalnızca uyduların gördüğü yangınlarla çizilmiş; üstünde Terra, Aqua ve S-NPP yörünge şeritleri");
  for (let lo = -150; lo <= 150; lo += 30) E("line", { x1: x(lo), x2: x(lo), y1: 0, y2: h, stroke: css("--line"), "stroke-width": .6, opacity: .55 }, s);
  for (let la = -30; la <= 60; la += 30) E("line", { x1: 0, x2: w, y1: y(la), y2: y(la), stroke: css("--line"), "stroke-width": la === 0 ? 1 : .6, opacity: .55 }, s);
  if (H && HI.v) {
    /* 0,25 derecelik hucreler (Hesapla haritasiyla ayni): tek resim olarak ciz, keskin kenarli */
    const W = HI.w, r0 = (LA0 + 90) * 4, r1 = (LA1 + 90) * 4, rows = r1 - r0, cv = document.createElement("canvas"); cv.width = W; cv.height = rows;
    const c2 = cv.getContext("2d"), im = c2.createImageData(W, rows), b = new Uint32Array(im.data.buffer), lut = new Uint32Array(256);
    for (let q = 1; q < 256; q++) { const m = heat(.12 + .88 * (q - 1) / 254).match(/\d+/g).map(Number); lut[q] = (255 << 24 | m[2] << 16 | m[1] << 8 | m[0]) >>> 0; }
    for (let r = 0; r < rows; r++) { const src = (r1 - 1 - r) * W, dst = r * W; for (let c = 0; c < W; c++) { const q = HI.v[src + c]; if (q) b[dst + c] = lut[q]; } }
    c2.putImageData(im, 0, 0);
    E("image", { href: cv.toDataURL(), x: 0, y: 0, width: w, height: h, preserveAspectRatio: "none", style: "image-rendering:pixelated" }, s);
  } else if (H) {
    const mx = Math.log1p(Math.max(...H.v)), cw = w / 360, ch = h / (LA1 - LA0), g = E("g", {}, s);
    for (let k = 0; k < H.v.length; k++) {
      const la = H.lat[k], lo = H.lon[k]; if (la < LA0 || la >= LA1) continue;
      E("rect", { x: x(lo), y: y(la + 1), width: cw + .3, height: ch + .3, fill: heat(.12 + .88 * Math.log1p(H.v[k]) / mx) }, g);
    }
  }
  if (H) {
    const key = document.getElementById("herokey");
    key.innerHTML = `<i style="background:linear-gradient(90deg,${["--t0","--t1","--t2","--t3","--t4"].map(css).join(",")})"></i><span>${H.src === "dünya" ? "Haritayı " + H.y0 + "–" + H.y1 + " arasında VIIRS'in gördüğü yangınlar çiziyor." : "İndirilen bölgelerde " + H.y0 + "–" + H.y1 + " arası yangın tespiti."} Şeritler: gündüz geçişleri, gerçek tarama genişliğinde.</span>`;
  }
  const dr = D.drift24 || D.drift, last = a => a ? a.filter(v => v != null).slice(-1)[0] : null;
  /* ekvator gecisleri: Terra ~40B, Aqua ~60D, S-NPP ~140D (temsili; seritlerin genisligi gercek tarama genisligi) */
  const sats = [["Terra", css("--terra"), 98.2, 98.9, 146.2, 2330, last(dr && dr.terra), true], ["Aqua", css("--aqua"), 98.2, 98.9, 66.2, 2330, last(dr && dr.aqua), false],
                ["S-NPP", css("--snpp"), 98.7, 101.4, 146.4, 3060, last(dr && dr.snpp), false]];
  const anim = !HERO_DONE && !matchMedia("(prefers-reduced-motion: reduce)").matches;
  sats.forEach(([nm, col, inc, per, lon0, swath, lt, desc], si) => {
    const segs = groundTrack(inc, per, lon0, desc), sw = swath / 111.32 / 360 * w, gg = E("g", {}, s);
    segs.forEach(seg => {
      if (seg.length < 2) return;
      const pts = seg.map(([lo, la]) => `${x(lo).toFixed(1)},${y(la).toFixed(1)}`).join(" ");
      const band = E("polyline", { points: pts, fill: "none", stroke: col, "stroke-width": sw, "stroke-opacity": .1, "stroke-linejoin": "round", "stroke-linecap": "butt" }, gg);
      const ln = E("polyline", { points: pts, fill: "none", stroke: col, "stroke-width": 1.6, "stroke-opacity": .9 }, gg);
      if (anim) [band, ln].forEach(el => { const L = el.getTotalLength(); el.style.strokeDasharray = L; el.style.strokeDashoffset = L;
        el.style.animation = `draw 2.8s cubic-bezier(.3,.6,.2,1) ${si * .35}s forwards`; });
    });
    const seg = segs.find(sg => sg.some(([lo, la]) => la > 40 && la < 56)) || segs[0];
    const pt = seg.find(([lo, la]) => la > 40 && la < 56) || seg[0];
    const lx = x(pt[0]), ly = y(pt[1]);
    const t1 = T(s, lx + 10, ly, nm + (lt != null ? " " + ft(lt) : ""), { class: "lab" }); t1.style.fill = col; t1.style.fontSize = "13px";
    t1.style.stroke = css("--ground"); t1.style.strokeWidth = "5px"; t1.style.paintOrder = "stroke"; t1.style.strokeLinejoin = "round"; t1.style.fontWeight = "600";
    if (anim) { t1.style.opacity = 0; t1.style.animation = `rise .6s ease ${1.6 + si * .35}s forwards`; }
  });
  HERO_DONE = true;
}
function passTable() {
  const R = D.drift24 || D.drift, t = document.getElementById("passtab");
  if (!R) { t.innerHTML = ""; return; }
  const yrs = R.years, pick = arr => { if (!arr) return [null, null, null, null]; let i0 = arr.findIndex(v => v != null), i1 = arr.length - 1 - [...arr].reverse().findIndex(v => v != null);
    return i0 < 0 ? [null, null, null, null] : [yrs[i0], arr[i0], yrs[i1], arr[i1]]; };
  const rows = [["Terra", css("--terra"), R.terra, "1999", "Yakıtı bitti, veri toplama Ocak 2027'de sona eriyor."],
                ["Aqua", css("--aqua"), R.aqua, "2002", "Yörüngesi kayıyor, 2027 sonbaharına kadar çalışacak."],
                ["S-NPP", css("--snpp"), R.snpp, "2011", "VIIRS taşıyor. FIRMS arşivinde verisi 2 Kasım 2026'ya kadar."],
                ["NOAA-20", css("--snpp"), R.noaa20, "2017", "VIIRS taşıyor, geçiş saati sabit."]];
  t.innerHTML = "<tr><th>Uydu</th><th>İlk yıl</th><th>Son yıl</th></tr>" + rows.map(([nm, col, arr, ly, note]) => {
    const [y0, v0, y1, v1] = pick(arr); if (v0 == null) return "";
    const moved = Math.abs(v1 - v0) >= .4;
    return `<tr><td class="sat"><i style="background:${col}"></i>${nm}<small>${ly}'de fırlatıldı. ${note}</small></td><td>${ft(v0)}<small>${y0}</small></td><td class="${moved ? "moved" : ""}">${ft(v1)}<small>${y1}</small></td></tr>`;
  }).join("");
}
function shiftCard() {
  /* Yorunge kaymasi: uydularin gecis saatinin ilk ve son yili bir gun cizelgesinde + uc sayi. Hepsi veriden hesaplanir. */
  const sec = document.getElementById("shiftsec"), R = D.drift24 || D.drift, W = D.driftWorld;
  if (!R || !R.terra) { sec.hidden = true; return; }
  sec.hidden = false;
  const first = arr => { const i = (arr || []).findIndex(v => v != null); return i < 0 ? null : [R.years[i], arr[i]]; };
  const lastv = arr => { const a = arr || []; for (let i = a.length - 1; i >= 0; i--) if (a[i] != null) return [R.years[i], a[i]]; return null; };
  const rows = [["Terra", css("--terra"), R.terra], ["Aqua", css("--aqua"), R.aqua], ["S-NPP / NOAA-20", css("--snpp"), R.noaa20 && lastv(R.noaa20) ? R.noaa20 : R.snpp]]
    .map(([n, c, a]) => [n, c, first(a), lastv(a)]).filter(r => r[2] && r[3]);
  /* gun cizelgesi: 08:00-16:00 yerel gunes saati */
  const box = document.getElementById("shiftsvg"); box.innerHTML = "";
  const w = 620, rh = 46, top = 22, h = top + rows.length * rh + 26, H0 = 8, H1 = 17, L = 96, Rr = 8;
  const x = t => L + (t - H0) / (H1 - H0) * (w - L - Rr);
  const s = E("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": "Uyduların yerel geçiş saati, ilk ve son yıl" }, box);
  E("rect", { x: x(14), y: top - 10, width: x(15) - x(14), height: rows.length * rh + 4, fill: css("--peak") }, s);
  T(s, x(15) + 6, top + rows.length * rh - 12, "yangın tepesi", {}).style.fill = css("--harm");
  for (let t = H0; t <= H1; t += 2) {
    E("line", { x1: x(t), x2: x(t), y1: top - 6, y2: top + rows.length * rh - 4, stroke: css("--line"), "stroke-width": 1 }, s);
    T(s, x(t), h - 6, String(t).padStart(2, "0") + ":00", { "text-anchor": "middle" });
  }
  rows.forEach(([nm, col, [y0, v0], [y1, v1]], i) => {
    const cy = top + i * rh + rh / 2 - 4, moved = Math.abs(v1 - v0) >= .4;
    T(s, 0, cy + 6, nm.split(" / ")[0], { class: "sat" }).style.fill = col;
    if (moved) {
      E("line", { x1: x(v0), x2: x(v1) + (v1 > v0 ? -9 : 9), y1: cy, y2: cy, stroke: col, "stroke-width": 2.5 }, s);
      const d = v1 > v0 ? 1 : -1, ax = x(v1) - d * 9;
      E("path", { d: `M${ax},${cy - 6} L${x(v1) - d * 1},${cy} L${ax},${cy + 6} Z`, fill: col }, s);
      E("circle", { cx: x(v0), cy, r: 5, fill: css("--ground"), stroke: col, "stroke-width": 2 }, s);
      T(s, x(v0), cy - 12, `${ft(v0)} · ${y0}`, { "text-anchor": "middle" });
      T(s, x(v1), cy + 22, `${ft(v1)} · ${y1}`, { "text-anchor": "middle" }).style.fill = col;
    } else {
      E("circle", { cx: x(v1), cy, r: 5, fill: col }, s);
      T(s, x(v1), cy - 12, `${ft(v1)} · sabit`, { "text-anchor": "middle" });
    }
  });
  document.getElementById("shiftcap").textContent = "Türkiye üzerinden gündüz geçişinin yerel güneş saati, yıllık medyan (FIRMS tespitlerinden). Boş daire ilk yıl, ok ucu son yıl. Kırmızı dilim gün içi yangın tepesi (14:00–15:00). VIIRS uyduları aynı saatte kalıyor.";
  /* sayilar */
  const nums = document.getElementById("shiftnums"), tr = (D.evidence || []).find(e => e.key === "TR");
  const tf = first(R.terra), tl = lastv(R.terra);
  let html = `<div><dt>Terra'nın geçiş saati</dt><dd><b>${ft(tf[1])} → ${ft(tl[1])}</b><span>${tf[0]} → ${tl[0]} arasında ${nf(Math.abs(tl[1] - tf[1]) * 60)} dakika sabaha kaydı. Yangınlar henüz başlamadan geçiyor.</span></dd></div>`;
  if (W && W.terra && W.terra.length > 1) {
    const i0 = W.terra.findIndex(v => v > 0), i1 = W.terra.length - 1, ch = (W.terra[i1] / W.terra[i0] - 1) * 100;
    html += `<div><dt>Terra'nın gördüğü yangın, dünya geneli</dt><dd><b class="hot">${ch < 0 ? "−" : "+"}%${nf(Math.abs(ch))}</b><span>S-NPP'nin her 1.000 tespitine karşı: ${W.years[i0]} yılında ${nf(W.terra[i0] * 1000)}, ${W.years[i1]} yılında ${nf(W.terra[i1] * 1000)}.</span></dd></div>`;
  }
  if (tr && tr.kayma) {
    const ks = Object.entries(tr.kayma), [ly, lv] = ks[ks.length - 1];
    html += `<div><dt>Türkiye'de ham MODIS, uyumlu seriye göre</dt><dd><b class="hot">+%${nf(lv - 100)}</b><span>${ly} yılında olduğundan fazla sayıyor: Aqua öğleden sonraki yangın tepesine kaydı.</span></dd></div>`;
  }
  nums.innerHTML = html + `<p>Uyumlu seri VIIRS'e dayandığı için bu kaymadan etkilenmiyor.</p>`;
}
function homeText() {
  const ev = D.evidence || [], tr = ev.find(e => e.key === "TR"), hi = ev.filter(e => e.guven === "yüksek" && e.isabet);
  let txt = "";
  if (tr && tr.isabet) txt += `Türkiye'de dönüşüm ${tr.egitim} yıllarından öğrenildi; modelin hiç görmediği ${tr.test} verisinde yıllık toplamı %${tr.isabet} isabetle, günlük seyri r = ${nf(tr.rd, 2)} ile tutturdu. `;
  if (hi.length > 1) txt += `Güveni yüksek ${hi.length} bölgede test isabeti %${Math.min(...hi.map(e => e.isabet))} ile %${Math.max(...hi.map(e => e.isabet))} arasında. `;
  if (tr && tr.kayma) { const ks = Object.entries(tr.kayma); const [ly, lv] = ks[ks.length - 1]; txt += `2021'den sonra ham MODIS, uyumlu seriden her yıl daha fazla ayrılıyor: ${ly} yılında %${lv - 100} fazla sayıyor, çünkü Aqua'nın geçişi öğleden sonraki yangın tepesine kaydı.`; }
  document.getElementById("proof").textContent = txt;
  const st = document.getElementById("status");
  st.textContent = D.lastDay ? `Veri ${fdl(D.lastDay)} tarihine kadar` : "";
  const ad = document.getElementById("aboutdata");
  if (ad) ad.textContent = ev.length ? `${ev.length} bölge: ` + ev.map(e => `${e.ad} (${e.y0}–${e.y1})`).join(", ") + "." + (D.hero && D.hero.src === "dünya" ? ` Ayrıca tüm dünya, ${D.hero.y0}–${D.hero.y1}.` : "") : "";
}

/* ---------- 18 bolgede test ---------- */
function evidence() {
  const t = document.getElementById("evtab"), ev = D.evidence || [];
  if (!ev.length) { t.innerHTML = "<tr><td>Henüz bölge veri seti yok.</td></tr>"; return; }
  t.innerHTML = "<tr><th>Bölge</th><th>Yıllar</th><th>Güven</th><th class='r'>İsabet</th><th class='r'>r gün</th><th class='r'>r hafta</th><th>En yoğun yıl</th></tr>" +
    ev.map(e => `<tr><td>${esc(e.ad)}</td><td>${e.y0}–${e.y1}<br><small>eğitim ${e.egitim || "–"}, test ${e.test || "–"}</small></td><td><span class="g g-${e.guven}">${e.guven}</span></td>` +
      `<td class="r">${e.isabet != null ? "%" + e.isabet : "–"}</td><td class="r">${e.rd != null ? nf(e.rd, 2) : "–"}</td><td class="r">${e.rw != null ? nf(e.rw, 2) : "–"}</td>` +
      `<td>${e.buyuk ? `${e.buyuk.y}<small> (normalin ${nf(e.buyuk.kat, 1)} katı)</small>` : "–"}</td></tr>`).join("");
}

/* ---------- gunluk seyir ---------- */
function line() {
  const w = W("line", 300), h = 300, L = 44, R = w < 520 ? 16 : 168, top = 14, bot = 34;
  const s = canvas("line", w, h), n = D.days.length;
  const vd = D.viirs.map(v => v / 10);
  const ym = nice(Math.max(...D.modis, ...D.harm, ...vd, 1));
  const x = i => L + i * (w - L - R) / (n - 1), y = v => top + (h - top - bot) * (1 - v / ym);
  for (let k = 0; k <= 4; k++) { const v = ym * k / 4; E("line", { x1: L, x2: w - R, y1: y(v), y2: y(v), class: "grid" }, s); T(s, L - 8, y(v) + 4, nf(v), { "text-anchor": "end" }); }
  D.days.forEach((d, i) => { if (i % 7 === 0) T(s, x(i), h - 10, fd(d), { "text-anchor": "middle" }); });
  const path = (arr, col, wid, dash) => E("polyline", { points: arr.map((v, i) => `${x(i)},${y(v)}`).join(" "), fill: "none", stroke: col, "stroke-width": wid, "stroke-dasharray": dash || "", "stroke-linejoin": "round" }, s);
  path(vd, css("--viirs"), 1.5, "3 4");
  path(D.modis, css("--modis"), 2);
  path(D.harm, css("--harm"), 3);
  D.modis.forEach((v, i) => { const c = E("circle", { cx: x(i), cy: y(v), r: 3, fill: css("--modis") }, s); tip(c, `${fd(D.days[i])}<b>${nf(v)}</b>gerçek MODIS · çevrilmiş ${nf(D.harm[i], 1)}`); });
  if (R > 100) {
    const lastY = arr => y(arr[n - 1]);
    const lbl = [[D.harm, "çevrilmiş VIIRS", css("--harm")], [D.modis, "gerçek MODIS", css("--modis")], [vd, "VIIRS ham ÷ 10", css("--viirs")]].map(([a, t, c]) => ({ y: lastY(a), t, c }));
    lbl.sort((a, b) => a.y - b.y); for (let k = 1; k < lbl.length; k++) if (lbl[k].y - lbl[k - 1].y < 16) lbl[k].y = lbl[k - 1].y + 16;
    lbl.forEach(o => { const t = T(s, w - R + 10, o.y + 4, o.t); t.style.fill = o.c; });
  }
  document.getElementById("calhow").innerHTML = `Kalın kırmızı çizgi çevrilmiş VIIRS, kahverengi çizgi gerçek MODIS. İki çizgi aynı günlerde yükselip alçalıyor (r = ${nf(D.rDaily, 2)}). MODIS tespitlerinin %${nf(D.emptyShare)}'si VIIRS'in o gün boş gördüğü hücrelerde, yani iki geçiş arasında başlayıp sönen yangınlarda. Bu kısım ${fd(D.cut)} öncesinde öğrenilen tek bir katsayıyla telafi ediliyor (k = ${nf(D.k, 2)}).`;
}

/* ---------- uyum basamaklari ---------- */
function steps() {
  const Lb = D.ladder.label, V = D.ladder.mv, n = V.length;
  const w = W("steps", 300), narrow = w < 560, h = narrow ? 360 : 330, top = 34, base = h - (narrow ? 88 : 64), L = 30;
  const s = canvas("steps", w, h), cw = (w - L) / n, yv = v => base - (base - top) * v / 100;
  [0, 25, 50, 100].forEach(v => { E("line", { x1: L, x2: w, y1: yv(v), y2: yv(v), class: "grid" }, s); T(s, L - 6, yv(v) + 4, v, { "text-anchor": "end" }); });
  V.forEach((v, i) => {
    const x = L + i * cw + 6, bw = cw - 12, ok = v >= 75;
    const r = E("rect", { x, y: yv(v), width: bw, height: base - yv(v), fill: ok ? css("--harm") : "none", stroke: ok ? css("--harm") : css("--ink2"), "stroke-width": ok ? 0 : 1.5 }, s);
    tip(r, `${Lb[i][0]} · ${Lb[i][1]}<b>%${nf(v, 1)}</b>MODIS tespitlerinin VIIRS'te karşılığı olan kısmı` + (D.ladder.vm[i] != null ? ` · ters yön %${nf(D.ladder.vm[i], 1)}` : ""));
    const t = T(s, x + bw / 2, yv(v) - 8, "%" + nf(v, 0), { "text-anchor": "middle", class: "big" }); t.style.fontSize = narrow ? "17px" : "24px";
    const a = T(s, x + bw / 2, base + 18, Lb[i][0].split(" (")[0], { "text-anchor": "middle", class: "lab" });
    const b = T(s, x + bw / 2, base + 34, Lb[i][1], { "text-anchor": "middle" });
    if (narrow) { [a, b].forEach((el, j) => { el.setAttribute("transform", `rotate(-38 ${x + bw / 2} ${base + 14 + j * 14})`); el.setAttribute("text-anchor", "end"); el.setAttribute("y", base + 14 + j * 14); }); }
    else { a.style.fontSize = b.style.fontSize = cw < 130 ? "10px" : "11px"; }
  });
  E("line", { x1: L, x2: w, y1: yv(75), y2: yv(75), stroke: css("--harm"), "stroke-width": 1.5, "stroke-dasharray": "6 4" }, s);
  const t = T(s, L + 6, yv(75) - 6, "hedef %75", { "text-anchor": "start" }); t.style.fill = css("--harm");

  const R = D.corr, w2 = W("corr", 300), rh = 26, lab = Math.min(260, w2 * .45), h2 = R.label.length * rh + 34;
  const c = canvas("corr", w2, h2), xr = v => lab + (w2 - lab - 40) * v;
  [0, .5, 1].forEach(v => { E("line", { x1: xr(v), x2: xr(v), y1: 0, y2: h2 - 22, class: "grid" }, c); T(c, xr(v), h2 - 6, v === 0 ? "r = 0" : nf(v, 1), { "text-anchor": "middle" }); });
  R.label.forEach((l, i) => {
    const y = 14 + i * rh;
    T(c, 0, y + 4, l, { class: "lab" }).style.fontSize = w2 < 420 ? "9.5px" : "11px";
    E("line", { x1: xr(0), x2: xr(R.r[i]), y1: y, y2: y, stroke: css("--viirs"), "stroke-width": 2 }, c);
    const d = E("circle", { cx: xr(R.r[i]), cy: y, r: 5, fill: css("--viirs") }, c);
    tip(d, `${l}<b>r = ${nf(R.r[i], 2)}</b>MODIS ve VIIRS sayıları arasındaki korelasyon (log)`);
    T(c, xr(R.r[i]) + 10, y + 4, nf(R.r[i], 2), { class: "lab" });
  });
}

/* ---------- kalicilik ---------- */
function pers() {
  const P = D.pers, w = W("pers", 300), h = 300, top = 30, base = h - 56, L = 34, n = P.label.length, cw = (w - L) / n;
  const s = canvas("pers", w, h), ym = nice(Math.max(...P.rate, 5)), yv = v => base - (base - top) * v / ym;
  [0, ym / 2, ym].forEach(v => { E("line", { x1: L, x2: w, y1: yv(v), y2: yv(v), class: "grid" }, s); T(s, L - 6, yv(v) + 4, "%" + nf(v), { "text-anchor": "end" }); });
  const nmax = Math.max(...P.n);
  P.label.forEach((l, i) => {
    const x = L + i * cw + cw * .2, bw = cw * .6;
    const r = E("rect", { x, y: yv(P.rate[i]), width: bw, height: base - yv(P.rate[i]), fill: heat(.25 + .75 * i / (n - 1)) }, s);
    tip(r, `${l}<b>%${nf(P.rate[i], 1)}</b>${nf(P.n[i])} olay · ort. FRP ${nf(P.frp[i])} MW`);
    T(s, x + bw / 2, yv(P.rate[i]) - 8, "%" + nf(P.rate[i], 1), { "text-anchor": "middle", class: "lab" });
    T(s, x + bw / 2, base + 18, l, { "text-anchor": "middle", class: "lab" });
    T(s, x + bw / 2, base + 34, nf(P.n[i]) + " olay", { "text-anchor": "middle" });
    E("rect", { x, y: base + 42, width: Math.max(bw * P.n[i] / nmax, 1), height: 3, fill: css("--ink2") }, s);
  });
  const total = P.n.reduce((a, b) => a + b, 0);
  document.getElementById("persnote").textContent = `VIIRS'in gördüğü ${nf(total)} yangın olayının %${nf(D.shortShare)}'i en fazla iki geçiş (~1 saatten kısa) sürüyor. Bunlar büyük çoğunlukla anız yangınları ve MODIS geçişine kadar sönüyorlar. Bu yüzden tek piksel düzeyinde düşük uyum bir hata değil, bölgedeki yangınların kısa ömrünün sonucu.`;
}

/* ---------- gunes saati ---------- */
function dial() {
  if (!D.drift24 && !D.drift) { document.getElementById("driftsec").hidden = true; return; }
  const R = D.drift24 || D.drift;
  if (D.drift24) document.getElementById("driftnote").textContent = `Terra ve Aqua'nın yakıtı bitiyor ve yörüngeleri kayıyor. Güneş saati, uyduların Türkiye üzerinden geçtiği yıllık medyan yerel saati gösteriyor. Her halka bir yıl, içten dışa ${R.years[0]}'ten ${R.years[R.years.length - 1]}'ya.`;
  const w = Math.min(W("dial", 300), 760), h = Math.round(w * .6), cx = w / 2, cy = h - 26;
  const r0 = w * .13, r1 = w * .43, n = R.years.length, rr = i => r0 + (r1 - r0) * i / (n - 1);
  const ang = hr => Math.PI * (1 - (hr - 6) / 12), P = (hr, r) => [cx + r * Math.cos(ang(hr)), cy - r * Math.sin(ang(hr))];
  const s = canvas("dial", w, h);
  const wedge = (h0, h1, ra, rb) => { const [a0, b0] = P(h0, rb), [a1, b1] = P(h1, rb), [c1, d1] = P(h1, ra), [c0, d0] = P(h0, ra);
    return `M${a0},${b0} A${rb},${rb} 0 0 1 ${a1},${b1} L${c1},${d1} A${ra},${ra} 0 0 0 ${c0},${d0} Z`; };
  E("path", { d: wedge(14, 15, r0 - 6, r1 + 8), fill: css("--peak") }, s);
  R.years.forEach((y, i) => {
    E("path", { d: `M${cx - rr(i)},${cy} A${rr(i)},${rr(i)} 0 0 1 ${cx + rr(i)},${cy}`, fill: "none", class: "grid" }, s);
    if (i === 0 || i === n - 1 || i === Math.floor((n - 1) / 2)) T(s, cx - rr(i), cy + 16, String(y), { "text-anchor": "middle" }).style.fontSize = "10px";
  });
  for (let hr = 6; hr <= 18; hr++) {
    const [a, b] = P(hr, r1 + 6), [c, d] = P(hr, r1 + (hr % 3 === 0 ? 16 : 11));
    E("line", { x1: a, y1: b, x2: c, y2: d, stroke: css("--ink2"), "stroke-width": 1 }, s);
    if (hr % 3 === 0) { const [tx, ty] = P(hr, r1 + 30); T(s, tx, ty + 4, String(hr).padStart(2, "0") + ":00", { "text-anchor": "middle", class: "lab" }); }
  }
  const ser = [["Terra", R.terra, css("--terra")], ["Aqua", R.aqua, css("--aqua")], ["S-NPP", R.snpp, css("--snpp")]];
  ser.forEach(([name, arr, col]) => {
    const pts = arr.map((v, i) => v == null ? null : P(v, rr(i)));
    const ok = pts.filter(Boolean);
    E("polyline", { points: ok.map(p => p.join(",")).join(" "), fill: "none", stroke: col, "stroke-width": 2 }, s);
    pts.forEach((p, i) => { if (!p) return; const c = E("circle", { cx: p[0], cy: p[1], r: i === n - 1 ? 6 : 4, fill: col, stroke: css("--ground"), "stroke-width": 1.5 }, s);
      tip(c, `${name} · ${R.years[i]}<b>${ft(arr[i])}</b>gündüz geçişi, yerel güneş saati`); });
    const lastI = arr.length - 1 - [...arr].reverse().findIndex(v => v != null), lp = pts[lastI];
    if (lp) { const v = arr[lastI], mid = Math.abs(v - 12) < 1.5, right = v > 12;
      const t = mid ? T(s, lp[0] + 10, lp[1] - 8, `${name} ${ft(v)}`, { "text-anchor": "start" })
                    : T(s, lp[0] + (right ? -12 : 12), lp[1] + 20, `${name} ${ft(v)}`, { "text-anchor": right ? "end" : "start" });
      t.style.fill = col; t.style.fontWeight = 600; }
  });
  const [px, py] = P(14.5, r1 + 46);
  T(s, px + 4, py, "yangın tepesi", { "text-anchor": "start" }).style.fill = css("--harm");

  const w2 = W("ratio", 300), h2 = 230, L = 40, Rr = w2 < 480 ? 12 : 120, top = 14, bot = 30;
  const c = canvas("ratio", w2, h2), yrs = R.years, m = yrs.length;
  const WR = D.driftWorld, wi = yv => WR ? WR.years.indexOf(yv) : -1;
  const wT = yrs.map(yv => wi(yv) >= 0 ? WR.terra[wi(yv)] : null), wA = yrs.map(yv => wi(yv) >= 0 ? WR.aqua[wi(yv)] : null);
  const ym = nice(Math.max(...R.terraRatio.filter(v => v != null), ...R.aquaRatio.filter(v => v != null), ...wT.filter(v => v != null), ...wA.filter(v => v != null)));
  const x = i => L + i * (w2 - L - Rr) / (m - 1), y = v => top + (h2 - top - bot) * (1 - v / ym);
  [0, ym / 2, ym].forEach(v => { E("line", { x1: L, x2: w2 - Rr, y1: y(v), y2: y(v), class: "grid" }, c); T(c, L - 6, y(v) + 4, nf(v, 2), { "text-anchor": "end" }); });
  yrs.forEach((yv, i) => { if (m <= 12 || i % 3 === 0 || i === m - 1) T(c, x(i), h2 - 8, String(yv), { "text-anchor": "middle" }); });
  const series = [[R.terraRatio, "Terra, Türkiye", css("--terra"), ""], [R.aquaRatio, "Aqua, Türkiye", css("--aqua"), ""]];
  if (WR) series.push([wT, "Terra, dünya", css("--terra"), "5 4"], [wA, "Aqua, dünya", css("--aqua"), "5 4"]);
  const labs = [];
  series.forEach(([arr, name, col, dash]) => {
    E("polyline", { points: arr.map((v, i) => v == null ? "" : `${x(i)},${y(v)}`).join(" "), fill: "none", stroke: col, "stroke-width": dash ? 1.8 : 2, "stroke-dasharray": dash }, c);
    arr.forEach((v, i) => { if (v == null) return; const d = E("circle", { cx: x(i), cy: y(v), r: dash ? 2.5 : 3.5, fill: dash ? css("--ground") : col, stroke: col, "stroke-width": 1.5 }, c); tip(d, `${name}, ${yrs[i]}<b>${nf(v, 3)}</b>MODIS tespiti / S-NPP tespiti`); });
    const lv = [...arr].reverse().find(v => v != null);
    if (lv != null) labs.push([y(lv) + 4, name, col]);
  });
  if (Rr > 60) { labs.sort((a, b2) => a[0] - b2[0]); for (let i = 1; i < labs.length; i++) labs[i][0] = Math.max(labs[i][0], labs[i - 1][0] + 14);
    labs.forEach(([ly, name, col]) => { const t = T(c, w2 - Rr + 10, ly, name); t.style.fill = col; }); }
  const cap = document.createElement("p"); cap.className = "cap";
  let wtxt = "";
  if (WR) { const i0 = 0, i1 = WR.years.length - 1, ch = 100 * (WR.terra[i1] / WR.terra[i0] - 1);
    wtxt = ` Kesikli çizgiler dünyanın tamamı: Terra'nın oranı ${WR.years[i0]}'ten ${WR.years[i1]}'ya %${nf(Math.abs(ch), 0)} ${ch < 0 ? "düştü" : "arttı"}, çünkü geçişi sabaha, yangınlar başlamadan önceki saatlere kaydı.`; }
  cap.textContent = (D.drift24 ? "Yıllık MODIS tespit sayısının S-NPP'ye oranı, kalıcı kaynaklar çıkarıldıktan sonra. Türkiye'de 2021'den sonra iki MODIS uydusunun oranı da yükseliyor; ham MODIS bu yüzden uyumlu seriden giderek ayrılıyor." : "Gündüz tespit sayısının VIIRS'e oranı, kalıcı kaynaklar çıkarıldıktan sonra.") + wtxt;
  document.getElementById("ratio").appendChild(cap);
}


/* ================= KESIF ARACI ================= */
const DAY0 = Date.UTC(2000, 0, 1), MS = 864e5;
const VIIRS_START = Math.round((Date.UTC(2012, 0, 20) - DAY0) / MS);
const dDate = d => new Date(DAY0 + d * MS);
const dIso = d => dDate(d).toISOString().slice(0, 10);
const dYear = d => dDate(d).getUTCFullYear();
const yStart = y => Math.round((Date.UTC(y, 0, 1) - DAY0) / MS);
function makeF(cv) {
  const LX = cv.x.map(Math.log), LY = cv.y.map(v => Math.log(Math.max(v, 1e-6))), n = LX.length;
  const S_LO = (LY[1] - LY[0]) / (LX[1] - LX[0]);
  const S_HI = cv.hi != null ? cv.hi : (LY[n - 1] - LY[n - 2]) / (LX[n - 1] - LX[n - 2]);
  return v => {
    if (v <= 0) return 0;
    const lx = Math.log(v);
    if (lx <= LX[0]) return Math.exp(LY[0] + S_LO * (lx - LX[0]));
    if (lx >= LX[n - 1]) return Math.exp(LY[n - 1] + S_HI * (lx - LX[n - 1]));
    let i = 0; while (LX[i + 1] < lx) i++;
    const f = (lx - LX[i]) / (LX[i + 1] - LX[i]);
    return Math.exp(LY[i] + f * (LY[i + 1] - LY[i]));
  };
}
const fV = makeF(D.curve);
async function unpack(b64) {
  const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  if (typeof DecompressionStream === "undefined") throw new Error("Bu tarayıcı sıkıştırılmış veriyi açamıyor. Chrome, Edge ya da güncel Firefox ile aç.");
  const txt = await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
  return JSON.parse(txt);
}
const REGS = {};          // anahtar -> cozulmus bolge
let REG = null, SEL = null, Y0 = null, Y1 = null, MODE = "region", RES = null, UPLOADED = null;

function regYears(R) { let a = Infinity, b = -Infinity; for (const d of R.d) { if (d < a) a = d; if (d > b) b = d; } return [dYear(a), dYear(b), a, b]; }
function fillYears() {
  const [a, b] = regYears(REG), s0 = document.getElementById("exy0"), s1 = document.getElementById("exy1");
  s0.innerHTML = s1.innerHTML = "";
  for (let y = a; y <= b; y++) { s0.add(new Option(y, y)); s1.add(new Option(y, y)); }
  Y0 = a; Y1 = b; s0.value = a; s1.value = b;
}
async function useRegion(key) {
  const msg = document.getElementById("excal");
  try {
    if (key === "__upload") REG = UPLOADED;
    else { if (!REGS[key]) { msg.textContent = "veri açılıyor…"; REGS[key] = await unpack(D.regions.find(r => r.key === key).b64); } REG = REGS[key]; }
    if (REG && !REG.share && D.regions.some(r => r.key === key)) REG.share = { bolge: key };
    const sb = document.getElementById("exshare"); if (sb) sb.hidden = !(REG && REG.share);
    if (REG && REG.share && typeof shareSet === "function") shareSet(REG.share);
    SEL = null; fillYears(); exRender();
  } catch (e) { msg.textContent = e.message; msg.className = "msg err"; }
}

function compute() {
  const R = REG, d0 = Math.max(yStart(Y0), regYears(R)[2]), d1 = Math.min(yStart(Y1 + 1) - 1, regYears(R)[3]);
  const nd = d1 - d0 + 1, M = new Float64Array(nd), V = new Float64Array(nd), Ev = new Float64Array(nd);
  const nc = R.cells.lat.length, cE = new Float64Array(nc), cM = new Float64Array(nc), cV = new Float64Array(nc);
  const inSel = new Uint8Array(nc), F = R.f || (R.f = R.curve ? makeF(R.curve) : fV);
  for (let c = 0; c < nc; c++) inSel[c] = !SEL || (R.cells.lat[c] >= SEL.la0 && R.cells.lat[c] <= SEL.la1 && R.cells.lon[c] >= SEL.lo0 && R.cells.lon[c] <= SEL.lo1) ? 1 : 0;
  for (let i = 0; i < R.d.length; i++) {
    const d = R.d[i]; if (d < d0 || d > d1) continue;
    const c = R.c[i], m = R.m[i], v = R.v[i] / 10;
    const e = v > 0 ? R.k * F(v) : (d < VIIRS_START ? m : 0);
    cE[c] += e; cM[c] += m; cV[c] += v;
    if (!inSel[c]) continue;
    const j = d - d0; M[j] += m; V[j] += v; Ev[j] += e;
  }
  return { d0, nd, M, V, E: Ev, cE, cM, cV };
}

function geoRings(g) {
  if (!g) return [];
  if (g.type === "Polygon") return g.coordinates;
  if (g.type === "MultiPolygon") return g.coordinates.flat();
  return [];
}
function geoBBox(g) {
  let w = 999, s = 999, e = -999, n = -999;
  geoRings(g).forEach(rg => rg.forEach(([x, y]) => { if (x < w) w = x; if (x > e) e = x; if (y < s) s = y; if (y > n) n = y; }));
  return [w, s, e, n];
}
function lin(xs, ys) {
  const n = xs.length, mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; syy += (ys[i] - my) ** 2; }
  const b = sxx ? sxy / sxx : 0; return { b, a: my - b * mx, r: sxx && syy ? sxy / Math.sqrt(sxx * syy) : 0 };
}
function yearAxis(s, st, x, h, bot, rot) {
  st.forEach((r, i) => {
    if (rot) T(s, x(i) + 4, h - bot + 14, String(r.y), { "text-anchor": "end", transform: `rotate(-55 ${x(i) + 4} ${h - bot + 14})` });
    else T(s, x(i), h - 8, String(r.y), { "text-anchor": "middle" });
  });
}

function exMap() {
  const R = REG, C = R.cells, w = W("exmap", 300), layer = document.getElementById("exlayer").value;
  const vals = { e: RES.cE, m: RES.cM, v: RES.cV }[layer];
  const bb = R.boundary ? (R._bb || (R._bb = geoBBox(R.boundary))) : null;
  const lo0 = Math.floor(Math.min(...C.lon, bb ? bb[0] : 999) - .5), lo1 = Math.ceil(Math.max(...C.lon, bb ? bb[2] : -999) + .5),
        la0 = Math.floor(Math.min(...C.lat, bb ? bb[1] : 999) - .5), la1 = Math.ceil(Math.max(...C.lat, bb ? bb[3] : -999) + .5);
  const k = Math.cos(((la0 + la1) / 2) * Math.PI / 180), pad = 34;
  const sc = Math.min((w - pad - 12) / ((lo1 - lo0) * k), 560 / (la1 - la0)), h = Math.round((la1 - la0) * sc) + pad + 8;
  const X = lo => pad + (lo - lo0) * k * sc, Y = la => 8 + (la1 - la) * sc, iX = x => lo0 + (x - pad) / (k * sc), iY = y => la1 - (y - 8) / sc;
  const s = canvas("exmap", w, h);
  const span = lo1 - lo0, step = span > 12 ? 2 : span > 4 ? 1 : .5;
  for (let lo = Math.ceil(lo0 / step) * step; lo <= lo1; lo += step) { E("line", { x1: X(lo), x2: X(lo), y1: Y(la1), y2: Y(la0), class: "grid" }, s); T(s, X(lo), h - 8, nf(lo, step < 1 ? 1 : 0) + "°", { "text-anchor": "middle" }); }
  for (let la = Math.ceil(la0 / step) * step; la <= la1; la += step) { E("line", { x1: pad, x2: X(lo1), y1: Y(la), y2: Y(la), class: "grid" }, s); T(s, pad - 6, Y(la) + 4, nf(la, step < 1 ? 1 : 0) + "°", { "text-anchor": "end" }); }
  const mx = Math.log1p(Math.max(...vals, 1)), cs = R.cell * sc, g = E("g", {}, s);
  vals.forEach((v, i) => {
    if (v <= 0) return;
    const r = E("rect", { x: X(C.lon[i] - R.cell / 2), y: Y(C.lat[i] + R.cell / 2), width: Math.max(cs * k - .5, 1), height: Math.max(cs - .5, 1), fill: heat(.12 + .88 * Math.log1p(v) / mx) }, g);
    tip(r, `${nf(C.lat[i], 2)}° · ${nf(C.lon[i], 2)}°<b>${nf(v, 1)}</b>${{ e: "MODIS-eşdeğeri", m: "MODIS", v: "VIIRS (uydu ort.)" }[layer]} · ${Y0}–${Y1}`);
  });
  if (bb) { const d = geoRings(R.boundary).map(rg => "M" + rg.map(p => X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join("L") + "Z").join("");
    E("path", { d, fill: "none", stroke: css("--ink"), "stroke-width": 1.3, "stroke-linejoin": "round", opacity: .75 }, s); }
  (R.cities || []).forEach(c => {
    if (c.lon < lo0 || c.lon > lo1 || c.lat < la0 || c.lat > la1) return;
    E("circle", { cx: X(c.lon), cy: Y(c.lat), r: 2.5, fill: css("--ink") }, s);
    T(s, X(c.lon) + 6, Y(c.lat) - 5, c.n, { class: "lab" });
  });
  const box = E("rect", { fill: "none", stroke: css("--harm"), "stroke-width": 2, "stroke-dasharray": "5 3", visibility: "hidden" }, s);
  const showBox = b => { if (!b) { box.setAttribute("visibility", "hidden"); return; } box.setAttribute("visibility", "visible");
    box.setAttribute("x", X(b.lo0)); box.setAttribute("y", Y(b.la1)); box.setAttribute("width", Math.max(X(b.lo1) - X(b.lo0), 1)); box.setAttribute("height", Math.max(Y(b.la0) - Y(b.la1), 1)); };
  showBox(SEL);
  const pt = e => { const r = s.getBoundingClientRect(); return [(e.clientX - r.left) * w / r.width, (e.clientY - r.top) * h / r.height]; };
  let start = null;
  s.style.touchAction = "none";
  s.addEventListener("pointerdown", e => { start = pt(e); s.setPointerCapture(e.pointerId); });
  s.addEventListener("pointermove", e => { if (!start) return; const p = pt(e);
    showBox({ lo0: iX(Math.min(start[0], p[0])), lo1: iX(Math.max(start[0], p[0])), la0: iY(Math.max(start[1], p[1])), la1: iY(Math.min(start[1], p[1])) }); });
  s.addEventListener("pointerup", e => { if (!start) return; const p = pt(e), dx = Math.abs(p[0] - start[0]), dy = Math.abs(p[1] - start[1]);
    if (dx < 6 && dy < 6) { const lo = iX(p[0]), la = iY(p[1]); SEL = { lo0: lo - .5, lo1: lo + .5, la0: la - .5, la1: la + .5 }; }
    else SEL = { lo0: iX(Math.min(start[0], p[0])), lo1: iX(Math.max(start[0], p[0])), la0: iY(Math.max(start[1], p[1])), la1: iY(Math.min(start[1], p[1])) };
    start = null; exRender(); });
}

function seasonStats() {
  const out = [];
  for (let y = Y0; y <= Y1; y++) {
    const a = Math.max(yStart(y) - RES.d0, 0), b = Math.min(yStart(y + 1) - RES.d0, RES.nd);
    if (b <= a) continue;
    let tot = 0, totM = 0, totV = 0, act = 0;
    for (let j = a; j < b; j++) { tot += RES.E[j]; totM += RES.M[j]; totV += RES.V[j]; if (RES.E[j] > 0) act++; }
    const row = { y, tot, totM, totV, act, n: b - a, start: null, end: null, peak: null };
    if (tot > 0) {
      let c = 0; for (let j = a; j < b; j++) { c += RES.E[j]; if (row.start == null && c >= .1 * tot) row.start = j; if (c >= .9 * tot) { row.end = j; break; } }
      let best = -1, win = 0; const W2 = 7;
      for (let j = a; j < b; j++) { win = 0; for (let q = Math.max(a, j - W2); q <= Math.min(b - 1, j + W2); q++) win += RES.E[q]; if (win > best) { best = win; row.peak = j; } }
    }
    out.push(row);
  }
  return out;
}
const dShort = j => { const t = dDate(RES.d0 + j); return t.getUTCDate() + " " + AY[t.getUTCMonth()]; };

function exSummary(st) {
  const tot = st.reduce((a, r) => a + r.tot, 0);
  const area = SEL ? `${nf(SEL.la0, 1)}–${nf(SEL.la1, 1)}° enlem, ${nf(SEL.lo0, 1)}–${nf(SEL.lo1, 1)}° boylam` : `${REG.name} (tüm alan)`;
  const mon = new Float64Array(12); for (let j = 0; j < RES.nd; j++) mon[dDate(RES.d0 + j).getUTCMonth()] += RES.E[j];
  const top = [...mon.keys()].sort((a, b) => mon[b] - mon[a]).slice(0, 2).filter(m => mon[m] > 0).map(m => ["Ocak","Şubat","Mart","Nisan","Mayıs","Haziran","Temmuz","Ağustos","Eylül","Ekim","Kasım","Aralık"][m]);
  const peaks = st.filter(r => r.peak != null);
  document.getElementById("exsum").innerHTML = tot > 0
    ? `${area}, ${Y0 === Y1 ? Y0 : Y0 + "–" + Y1}: uyumlulaştırılmış toplam <span class="n">${nf(tot)}</span> MODIS-eşdeğeri tespit. En yoğun aylar <span class="n">${top.join(" ve ")}</span>.` + (peaks.length > 1 ? ` Sezon tepesi yıllar arasında ${dShort(Math.min(...peaks.map(r => r.peak - (yStart(r.y) - RES.d0))))}–${dShort(Math.max(...peaks.map(r => r.peak - (yStart(r.y) - RES.d0))))} aralığında gezdi.` : "")
    : `${area}, ${Y0}–${Y1}: bu alanda ve dönemde tespit yok. Haritada başka bir alan seç ya da yıl aralığını genişlet.`;
  const cal = document.getElementById("excal");
  cal.className = "msg";
  cal.innerHTML = guvenBox(REG) + `Kalibrasyon katsayısı k = ${nf(REG.k, 2)}. ${REG.calib}` + (REG.note ? `<br>${REG.note}` : "");
}
function guvenBox(R) {
  /* Guven etiketini kullaniciya sade dille acikla. Sunucu sonuclarinda R.guven/R.unc var; hazir bolgelerde kanit tablosundan al. */
  const ev = (D.evidence || []).find(e => e.key === R.key);
  const g = R.guven || (ev && ev.guven), u = R.unc != null ? R.unc : (ev && ev.unc);
  if (!g) return "";
  const why = {
    "yüksek": "Kalibrasyon yıllarında bol MODIS tespiti var ve yıllık toplamlar az sapıyor. Yıllık ve aylık sonuçlar güvenilir.",
    "orta": "Tespit sayısı ya da yıllar arası sapma orta düzeyde. Eğilimler ve sezon güvenilir; tek tek yılları dikkatle yorumla.",
    "düşük": "Kalibrasyon için az MODIS tespiti var ya da yıllar arası sapma büyük. Sonuç yön gösterir; kesin sayı olarak kullanma."
  }[g] || "";
  return `<span class="gbox"><span class="g g-${g}">Güven: ${g}</span> ${why}${u ? ` Yıllık toplamda tipik sapma ±%${nf(100 * (Math.exp(u) - 1))}.` : ""}</span>`;
}

function exGrid() {
  const years = []; for (let y = Y0; y <= Y1; y++) years.push(y);
  const days = RES.nd;
  if (years.length === 1 || days <= 400) {
    const w = Math.max(W("exgrid", 640), 640), lab = 190, n = days, cw = (w - lab) / n, rh = 26, gap = 6;
    const rows = [["MODIS (gerçek)", RES.M], ["VIIRS ham ÷ 10", Array.from(RES.V, v => v / 10)], ["uyumlulaştırılmış", RES.E]];
    const h = 10 + rows.length * (rh + gap) + 34, s = canvas("exgrid", w, h);
    const mx = Math.log1p(Math.max(...RES.M, ...RES.E, 1));
    rows.forEach(([nm, arr], r) => { const y = 6 + r * (rh + gap); T(s, 0, y + rh / 2 + 4, nm, { class: r === 2 ? "lab" : "" });
      for (let j = 0; j < n; j++) { const v = arr[j], rc = E("rect", { x: lab + j * cw, y, width: Math.max(cw - (cw > 4 ? 1.5 : 0), .8), height: rh, fill: v > 0 ? heat(Math.log1p(v) / mx) : css("--cell0") }, s);
        if (v > 0) tip(rc, `${nm}<b>${nf(v, 1)}</b>${dIso(RES.d0 + j)}`); } });
    const yb = 6 + rows.length * (rh + gap) + 4;
    for (let j = 0; j < n; j++) { const t = dDate(RES.d0 + j); if (t.getUTCDate() === 1) { E("line", { x1: lab + j * cw, x2: lab + j * cw, y1: yb, y2: yb + 6, class: "grid" }, s); T(s, lab + j * cw + 3, yb + 18, AY[t.getUTCMonth()] + (t.getUTCMonth() === 0 ? " " + t.getUTCFullYear() : "")); } }
    document.getElementById("excalh").textContent = "Yanma takvimi, gün gün";
    return;
  }
  const w = Math.max(W("exgrid", 900), 900), lab = 56, cw = (w - lab) / 366, rh = years.length > 16 ? 13 : 17, gap = 3;
  const h = 8 + years.length * (rh + gap) + 34, s = canvas("exgrid", w, h);
  const mx = Math.log1p(Math.max(...RES.E, 1));
  years.forEach((y, r) => {
    const yy = 6 + r * (rh + gap); T(s, 0, yy + rh - 3, String(y), { class: y === 2012 ? "lab" : "" });
    const a = yStart(y) - RES.d0, b = yStart(y + 1) - RES.d0;
    for (let j = Math.max(a, 0); j < Math.min(b, RES.nd); j++) {
      const v = RES.E[j], rc = E("rect", { x: lab + (j - a) * cw, y: yy, width: Math.max(cw - .25, .6), height: rh, fill: v > 0 ? heat(Math.log1p(v) / mx) : css("--cell0") }, s);
      if (v > 0) tip(rc, `${dIso(RES.d0 + j)}<b>${nf(v, 1)}</b>MODIS-eşdeğeri · MODIS ${nf(RES.M[j])} · VIIRS ${nf(RES.V[j], 1)}`);
    }
    if (y === 2012 && years.length > 1) { E("line", { x1: lab - 6, x2: w, y1: yy - 1.5, y2: yy - 1.5, stroke: css("--viirs"), "stroke-width": 1.5, "stroke-dasharray": "4 3" }, s); }
  });
  const yb = 6 + years.length * (rh + gap) + 4;
  [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334].forEach((d, m) => { E("line", { x1: lab + d * cw, x2: lab + d * cw, y1: yb, y2: yb + 6, class: "grid" }, s); T(s, lab + d * cw + 3, yb + 18, AY[m]); });
  document.getElementById("excalh").textContent = "Yanma takvimi" + (years.includes(2012) ? " (kesikli çizgi: VIIRS dönemi başlıyor)" : "");
}

function exYear(st) {
  const n = st.length, w = W("exyear", 300), rot = n > 1 && (w - 62) / (n - 1) < 36;
  const h = rot ? 286 : 260, L = 50, R = 14, top = 18, bot = rot ? 52 : 30;
  const s = canvas("exyear", w, h);
  if (!n) return;
  const u = REG.unc || 0, inBand = r => u > 0 && r.y >= 2012 && r.totV > 0, z = 1.28;
  const ym = nice(Math.max(...st.map(r => Math.max(inBand(r) ? r.tot * Math.exp(z * u) : r.tot, r.totM, r.totV / 10)), 1));
  const x = i => n === 1 ? (L + w - R) / 2 : L + i * (w - L - R) / (n - 1), y = v => top + (h - top - bot) * (1 - v / ym);
  [0, ym / 2, ym].forEach(v => { E("line", { x1: L, x2: w - R, y1: y(v), y2: y(v), class: "grid" }, s); T(s, L - 6, y(v) + 4, nf(v), { "text-anchor": "end" }); });
  yearAxis(s, st, x, h, bot, rot);
  const bs = st.map((r, i) => [r, i]).filter(([r]) => inBand(r));
  if (bs.length > 1) {
    const up = bs.map(([r, i]) => `${x(i)},${y(r.tot * Math.exp(z * u))}`), dn = bs.slice().reverse().map(([r, i]) => `${x(i)},${y(r.tot * Math.exp(-z * u))}`);
    E("polygon", { points: up.concat(dn).join(" "), fill: css("--harm"), "fill-opacity": .13 }, s);
  }
  const mean = st.reduce((a, r) => a + r.tot, 0) / n;
  if (n > 2) { E("line", { x1: L, x2: w - R, y1: y(mean), y2: y(mean), stroke: css("--ink2"), "stroke-dasharray": "1 4", "stroke-width": 1.5 }, s);
    T(s, w - R, y(mean) - 6, "ortalama", { "text-anchor": "end" }); }
  const vi = st.findIndex(r => r.y >= 2012);
  if (vi > 0) { E("line", { x1: x(vi), x2: x(vi), y1: top, y2: h - bot, stroke: css("--viirs"), "stroke-dasharray": "4 3" }, s); T(s, x(vi) + 4, top + 10, "VIIRS"); }
  const line = (key, col, wid, dash, f) => { const pts = st.map((r, i) => `${x(i)},${y(f ? f(r) : r[key])}`).join(" "); E("polyline", { points: pts, fill: "none", stroke: col, "stroke-width": wid, "stroke-dasharray": dash || "" }, s); };
  line(null, css("--viirs"), 1.5, "3 4", r => r.totV / 10);
  line("totM", css("--modis"), 2);
  line("tot", css("--harm"), 3);
  st.forEach((r, i) => { const c = E("circle", { cx: x(i), cy: y(r.tot), r: 4, fill: css("--harm") }, s);
    tip(c, `${r.y}<b>${nf(r.tot)}</b>uyumlulaştırılmış` + (inBand(r) ? ` (%80 aralık ${nf(r.tot * Math.exp(-z * u))}–${nf(r.tot * Math.exp(z * u))})` : "") + ` · MODIS ${nf(r.totM)} · VIIRS ham ${nf(r.totV)}`); });
  if (n > 2) { const im = st.reduce((bi, r, i) => r.tot > st[bi].tot ? i : bi, 0); T(s, x(im), y(st[im].tot) - 11, "rekor", { "text-anchor": "middle", class: "lab" }); }
  const ov = st.filter(r => r.y >= 2012 && r.totM > 0);
  const ratio = ov.length ? ov.reduce((a, r) => a + r.tot, 0) / ov.reduce((a, r) => a + r.totM, 0) : null;
  document.getElementById("exyearcap").innerHTML = `<span style="color:${css("--harm")}">kırmızı</span>: uyumlulaştırılmış · <span style="color:${css("--modis")}">kahverengi</span>: gerçek MODIS · <span style="color:${css("--viirs")}">mavi kesikli</span>: VIIRS ham ÷ 10.` +
    (bs.length > 1 ? ` Gölge: %80 belirsizlik aralığı (kalibrasyon yıllarındaki yıllık sapmadan, ±%${nf(100 * (Math.exp(z * u) - 1), 0)}).` : "") +
    (ratio ? ` İki sensörün birlikte olduğu yıllarda uyumlulaştırılmış toplam, gerçek MODIS'in ${nf(ratio, 2)} katı.` : "");
}

function exShift(st) {
  const n = st.length, w = W("exshift", 300), rot = n > 1 && (w - 62) / (n - 1) < 36;
  const h = rot ? 286 : 260, L = 50, R = 14, top = 12, bot = rot ? 52 : 30;
  const s = canvas("exshift", w, h), cap = document.getElementById("exshiftcap");
  const ok = st.filter(r => r.peak != null);
  if (ok.length < 2) { cap.textContent = "Sezon kayması için en az iki yıl seç."; return; }
  const doy = (r, j) => j - (yStart(r.y) - RES.d0);
  const x = i => n === 1 ? (L + w - R) / 2 : L + i * (w - L - R) / (n - 1), y = d => top + (h - top - bot) * d / 366;
  [0, 59, 120, 181, 243, 304, 365].forEach((d, k) => { E("line", { x1: L, x2: w - R, y1: y(d), y2: y(d), class: "grid" }, s);
    if (k < 6) T(s, L - 6, y(d) + 4, AY[[0, 2, 4, 6, 8, 10][k]], { "text-anchor": "end" }); });
  yearAxis(s, st, x, h, bot, rot);
  const xs = [], ps = [], ss = [];
  st.forEach((r, i) => { if (r.peak == null) return; const a = doy(r, r.start), e = doy(r, r.end), p = doy(r, r.peak);
    E("line", { x1: x(i), x2: x(i), y1: y(a), y2: y(e), stroke: css("--t2"), "stroke-width": Math.max(3, Math.min(9, (w - L - R) / n * .45)), "stroke-linecap": "round", "stroke-opacity": .5 }, s);
    const c = E("circle", { cx: x(i), cy: y(p), r: 4, fill: css("--harm") }, s);
    tip(c, `${r.y}<b>${dShort(r.peak)}</b>sezon tepesi · başı ${dShort(r.start)} · sonu ${dShort(r.end)}`);
    xs.push(r.y); ps.push(p); ss.push(a); });
  let txt = "Çubuk: sezonun %10–%90 aralığı, nokta: en yoğun 15 gün.";
  if (xs.length >= 5) {
    const fp = lin(xs, ps), fs = lin(xs, ss), xi = yr => x(st.findIndex(r => r.y === yr));
    E("line", { x1: xi(xs[0]), x2: xi(xs[xs.length - 1]), y1: y(fp.a + fp.b * xs[0]), y2: y(fp.a + fp.b * xs[xs.length - 1]), stroke: css("--ink"), "stroke-width": 1.5, "stroke-dasharray": "6 4" }, s);
    const say = (f, what) => { const d = f.b * 10; return Math.abs(d) < 3 || Math.abs(f.r) < .3 ? `${what} belirgin kaymadı` : `${what} her 10 yılda yaklaşık ${nf(Math.abs(d), 0)} gün ${d < 0 ? "erkene" : "geçe"} kaydı (r = ${nf(f.r, 2)})`; };
    txt += ` Kesikli çizgi tepenin eğilimi: ${say(fp, "tepe")}; ${say(fs, "sezon başı")}.`;
  } else txt += " Eğilim için en az 5 yıl seç.";
  cap.textContent = txt;
}

function exAnom(st) {
  const n = st.length, w = W("exanom", 300), rot = n > 1 && (w - 62) / n < 36;
  const h = rot ? 286 : 260, L = 50, R = 14, top = 16, bot = rot ? 52 : 30;
  const s = canvas("exanom", w, h), cap = document.getElementById("exanomcap");
  if (n < 2) { cap.textContent = "Sapma için en az iki yıl seç."; return; }
  const mean = st.reduce((a, r) => a + r.tot, 0) / n;
  if (mean <= 0) { cap.textContent = ""; return; }
  const dv = st.map(r => 100 * (r.tot - mean) / mean), m = Math.max(25, ...dv.map(Math.abs)), lim = Math.ceil(m / 25) * 25;
  const cw = (w - L - R) / n, x = i => L + i * cw + cw / 2, y = v => top + (h - top - bot) * (lim - v) / (2 * lim);
  [-lim, 0, lim].forEach(v => { E("line", { x1: L, x2: w - R, y1: y(v), y2: y(v), class: "grid" }, s); T(s, L - 6, y(v) + 4, v === 0 ? "0" : pct(v, true), { "text-anchor": "end" }); });
  yearAxis(s, st, x, h, bot, rot);
  st.forEach((r, i) => { const v = dv[i], rc = E("rect", { x: x(i) - cw * .34, width: cw * .68, y: Math.min(y(v), y(0)), height: Math.max(Math.abs(y(v) - y(0)), 1), fill: v >= 0 ? css("--harm") : css("--viirs") }, s);
    tip(rc, `${r.y}<b>${pct(v, true)}</b>${nf(r.tot)} · ortalama ${nf(mean)}`); });
  const hi = dv.indexOf(Math.max(...dv)), lo = dv.indexOf(Math.min(...dv));
  cap.textContent = `Her yılın uyumlulaştırılmış toplamı, seçilen yılların ortalamasına göre. En yoğun yıl ${st[hi].y} (${pct(dv[hi], true)}), en sakin yıl ${st[lo].y} (${pct(dv[lo], true)}).`;
}

function exMonth() {
  const w = W("exmonth", 300), h = 260, L = 46, top = 16, bot = 30, s = canvas("exmonth", w, h);
  const nY = Y1 - Y0 + 1, mon = new Float64Array(12);
  for (let j = 0; j < RES.nd; j++) mon[dDate(RES.d0 + j).getUTCMonth()] += RES.E[j] / nY;
  const ym = nice(Math.max(...mon, 1)), mean = mon.reduce((a, b) => a + b, 0) / 12, cw = (w - L) / 12, y = v => top + (h - top - bot) * (1 - v / ym);
  [0, ym / 2, ym].forEach(v => { E("line", { x1: L, x2: w, y1: y(v), y2: y(v), class: "grid" }, s); T(s, L - 6, y(v) + 4, nf(v), { "text-anchor": "end" }); });
  mon.forEach((v, m) => { const hi = v > mean, r = E("rect", { x: L + m * cw + cw * .15, y: y(v), width: cw * .7, height: h - bot - y(v), fill: hi ? css("--harm") : css("--line") }, s);
    tip(r, `${AY[m]}<b>${nf(v, 1)}</b>yıllık ortalama MODIS-eşdeğeri`); T(s, L + m * cw + cw / 2, h - 10, AY[m], { "text-anchor": "middle" }); });
}

function exTable(st) {
  const t = document.getElementById("extable");
  const head = "<tr><th>yıl</th><th class='r'>uyumlulaştırılmış</th><th class='r'>ortalamaya göre</th><th class='r'>MODIS</th><th class='r'>VIIRS ham</th><th>sezon başı</th><th>tepe</th><th>sezon sonu</th><th class='r'>aktif gün</th></tr>";
  const mean = st.length ? st.reduce((a, r) => a + r.tot, 0) / st.length : 0, dev = r => mean > 0 ? 100 * (r.tot - mean) / mean : 0;
  t.innerHTML = head + st.map(r => `<tr><td>${r.y}</td><td class='r'>${nf(r.tot)}</td><td class='r' style="color:${dev(r) >= 0 ? css("--harm") : css("--viirs")}">${pct(dev(r), true)}</td><td class='r'>${nf(r.totM)}</td><td class='r'>${nf(r.totV)}</td><td>${r.start != null ? dShort(r.start) : "–"}</td><td>${r.peak != null ? dShort(r.peak) : "–"}</td><td>${r.end != null ? dShort(r.end) : "–"}</td><td class='r'>${nf(r.act)}</td></tr>`).join("");
}

function exDownload() {
  if (!RES) return;
  const rows = ["tarih,modis_tespit,viirs_uydu_ortalamasi,uyumlulastirilmis_modis_esdegeri"];
  for (let j = 0; j < RES.nd; j++) rows.push(`${dIso(RES.d0 + j)},${RES.M[j]},${RES.V[j].toFixed(2)},${RES.E[j].toFixed(2)}`);
  const name = `ates_takvimi_${(REG.key || "veri").toLowerCase()}_${Y0}_${Y1}.csv`, text = rows.join("\n");
  const local = () => {
    const blob = new Blob([text], { type: "text/csv" }), a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  };
  if (EX_DL) { EX_DL.save({ filename: name, data: text }).catch(() => {}); return; }
  if (window.claude && typeof window.claude.use === "function") return; // artifact görünümü, indirme izni yok
  local();
}
let EX_DL = null;
if (window.claude && typeof window.claude.use === "function") {
  window.claude.use("downloads").then(d => { EX_DL = d; const b = document.getElementById("exdl"); if (b && !d) b.hidden = true; }).catch(() => {});
}

function exRender() {
  { const cb = document.getElementById("exclear"); if (cb) cb.hidden = !SEL; }
  if (!REG) return;
  Y0 = Math.min(+document.getElementById("exy0").value, +document.getElementById("exy1").value);
  Y1 = Math.max(+document.getElementById("exy0").value, +document.getElementById("exy1").value);
  RES = compute();
  const st = seasonStats();
  exMap(); exSummary(st); exGrid(); exYear(st); exMonth(); exShift(st); exAnom(st); exTable(st);
}

/* ---------- FIRMS CSV yukleme ---------- */
function parseFirms(text) {
  const lines = text.split(/\r?\n/).filter(l => l.trim());
  if (lines.length < 2) return [];
  const H = lines[0].split(",").map(h => h.trim().toLowerCase().replace(/^﻿/, ""));
  const ix = n => H.indexOf(n), iLat = ix("latitude"), iLon = ix("longitude"), iDate = ix("acq_date"), iSat = ix("satellite"), iIns = ix("instrument"), iType = ix("type"), iConf = ix("confidence");
  if (iLat < 0 || iLon < 0 || iDate < 0) throw new Error("Bu bir FIRMS dosyasına benzemiyor (latitude, longitude, acq_date sütunları yok).");
  const out = [];
  for (let k = 1; k < lines.length; k++) {
    const p = lines[k].split(",");
    const lat = +p[iLat], lon = +p[iLon], date = (p[iDate] || "").trim();
    if (!isFinite(lat) || !isFinite(lon) || !/^\d{4}-\d{2}-\d{2}/.test(date)) continue;
    const sat = iSat >= 0 ? (p[iSat] || "").trim() : "", ins = iIns >= 0 ? (p[iIns] || "").trim().toUpperCase() : "";
    const conf = iConf >= 0 ? (p[iConf] || "").trim() : "";
    const modis = ins ? ins.startsWith("MODIS") : (/^(terra|aqua|t|a)$/i.test(sat) || /^\d+$/.test(conf));
    const type = iType >= 0 ? (p[iType] || "").trim() : "";
    out.push({ lat, lon, d: Math.round((Date.UTC(+date.slice(0, 4), +date.slice(5, 7) - 1, +date.slice(8, 10)) - DAY0) / MS), modis, sat: modis ? "M" : (sat || "V"), type });
  }
  return out;
}
async function onFiles(files) {
  const msg = document.getElementById("upmsg"); msg.className = "msg";
  try {
    let pts = [];
    for (const f of files) pts = pts.concat(parseFirms(await f.text()));
    if (!pts.length) throw new Error("Dosyalarda okunabilir tespit bulunamadı.");
    const hasType = pts.some(p => p.type !== "");
    let dropped = 0;
    if (hasType) { const n0 = pts.length; pts = pts.filter(p => p.type === "" || p.type === "0"); dropped = n0 - pts.length; }
    else {
      const dmin = Math.min(...pts.map(p => p.d)), dmax = Math.max(...pts.map(p => p.d)), span = dmax - dmin + 1;
      const need = Math.max(4, Math.ceil(.5 * span)), seen = new Map();
      pts.forEach(p => { const key = Math.round(p.lat / .01) + "," + Math.round(p.lon / .01); if (!seen.has(key)) seen.set(key, new Set()); seen.get(key).add(p.d); });
      const n0 = pts.length; pts = pts.filter(p => seen.get(Math.round(p.lat / .01) + "," + Math.round(p.lon / .01)).size < need); dropped = n0 - pts.length;
    }
    const CELL = .25, cellIx = new Map(), lat = [], lon = [], agg = new Map(), satYear = new Map();
    pts.forEach(p => {
      const ci = Math.floor(p.lat / CELL), cj = Math.floor(p.lon / CELL), ck = ci + "," + cj;
      if (!cellIx.has(ck)) { cellIx.set(ck, lat.length); lat.push((ci + .5) * CELL); lon.push((cj + .5) * CELL); }
      const key = cellIx.get(ck) + ":" + p.d; if (!agg.has(key)) agg.set(key, { m: 0, vs: {} });
      const a = agg.get(key);
      if (p.modis) a.m++; else { a.vs[p.sat] = (a.vs[p.sat] || 0) + 1; const yr = dYear(p.d); if (!satYear.has(yr)) satYear.set(yr, new Set()); satYear.get(yr).add(p.sat); }
    });
    const c = [], d = [], m = [], v = [];
    agg.forEach((a, key) => { const [ci, dd] = key.split(":").map(Number), ns = (satYear.get(dYear(dd)) || new Set()).size || 1;
      const vs = Object.values(a.vs).reduce((x, y) => x + y, 0) / ns; c.push(ci); d.push(dd); m.push(a.m); v.push(Math.round(vs * 10)); });
    const nM = pts.filter(p => p.modis).length, nV = pts.length - nM;
    let k = D.kDefault, calib = `Türkiye verisinden öğrenilen varsayılan katsayı (dosyada MODIS ve VIIRS birlikte yok).`;
    if (nM && nV) {
      const order = [...new Set(d)].sort((a, b) => a - b), cut = order[Math.floor(order.length * .6)];
      let sm = 0, sf = 0, tm = 0, tf = 0;
      for (let i = 0; i < d.length; i++) { const f = fV(v[i] / 10); if (d[i] < cut) { sm += m[i]; sf += f; } else { tm += m[i]; tf += f; } }
      if (sf > 0 && sm > 0) { k = sm / sf; calib = `dosyanın ilk %60 gününde öğrenildi; kalan günlerde toplam isabet %${nf(100 * k * tf / Math.max(tm, 1))}.`; }
    }
    UPLOADED = { key: "YUKLENEN", name: `Yüklenen veri (${files.length} dosya)`, cell: CELL, k, calib, cells: { lat, lon }, c, d, m, v, cities: [],
      note: `${nf(nM)} MODIS ve ${nf(nV)} VIIRS tespiti okundu. ${nf(dropped)} tespit ${hasType ? "FIRMS'in 'type' alanına göre (volkan, sabit kaynak, deniz)" : "sürekli tekrar eden kaynak olarak"} ayıklandı.` };
    const sel = document.getElementById("exreg");
    if (![...sel.options].some(o => o.value === "__upload")) sel.add(new Option("Yüklenen veri", "__upload"));
    sel.value = "__upload";
    msg.textContent = UPLOADED.note;
    await useRegion("__upload");
  } catch (e) { msg.textContent = e.message; msg.className = "msg err"; }
}

/* ================= DUNYADA ARA (yerel sunucu) ================= */
let SERVER = false, WMAP = null, WLAYER = null, WSELP = null, WLEVEL = "il", WBUSY = false;
const $ = id => document.getElementById(id);
function wMsg(t, err) { const m = $("wmsg"); m.textContent = t || ""; m.className = "msg" + (err ? " err" : ""); }
/* ================= INTERNET SURUMU (sunucusuz hesaplama) =================
   Sayfa GitHub Pages gibi statik bir yerde acilip yaninda veri1/meta.json bulunursa (python web_veri.py),
   sunucu.py'nin hazir dunya verisi yolu (fast_region + payload) burada, tarayicida calisir.
   Yer arama dogrudan OpenStreetMap Nominatim'e gider. FIRMS'ten canli indirme yoktur. */
let WEB = null;
const WJOBS = {}, WTILES = new Map(), WNOM = new Map();
const W_DAY0 = Date.UTC(2000, 0, 1), W_CELL = 0.25, W_MAXTILE = 40;
const W_EDGES = [0, 0.4, 0.75, 1.1, 1.6, 2.2, 3.2, 4.5, 6.5, 9, 13, 18, 26, 36, 52, 75, 105, 150, 220, 320, 1e9];
const W_LEVEL_ZOOM = { ulke: 3, il: 5, ilce: 8, sehir: 10 };
let W_NOM_LAST = 0;
function tileUrl() { return WEB ? "https://tile.openstreetmap.org/{z}/{x}/{y}.png" : "/karo/{z}/{x}/{y}.png"; }
async function webProbe() {
  /* veri klasorlu (veri1/meta.json) ya da duz (meta.json, karolar ana dizinde) yuklenmis olabilir */
  for (const [path, flat] of [["veri1/meta.json", false], ["meta.json", true]]) {
    try { const r = await fetch(path, { cache: "no-cache" }); if (!r.ok) continue; const m = await r.json(); if (m && m.tiles) { m.flat = flat; return m; } }
    catch (e) {}
  }
  return null;
}
async function webApi(path, opt) {
  const u = new URL(path, location.href), q = u.searchParams, p = u.pathname;
  if (p.endsWith("/api/durum")) return { ok: true, web: true };
  if (p.endsWith("/api/veri_setleri")) return [{ key: "DUNYA", ad: "Dünya (tüm ülkeler)", olay: "", kutu: [-180, -90, 180, 90], y0: WEB.years[0], y1: WEB.years[WEB.years.length - 1] }];
  if (p.endsWith("/api/ara")) { const s = (q.get("q") || "").trim(); if (!s) throw new Error("arama metni boş"); return webSearch(s.slice(0, 200)); }
  if (p.endsWith("/api/ters")) return webReverse(+q.get("lat"), +q.get("lon"), q.get("duzey") || "il");
  if (p.endsWith("/api/bolge")) {
    const spec = JSON.parse((opt && opt.body) || "{}"), id = Math.random().toString(36).slice(2, 14);
    const job = WJOBS[id] = { durum: "çalışıyor", mesaj: "veri parçaları indiriliyor", yapilan: 0, toplam: 0, t0: performance.now(), sonuc: null };
    webRegion(spec, job).then(r => { job.sonuc = r; job.durum = "bitti"; }).catch(e => { console.error(e); job.durum = "hata"; job.mesaj = e.message || String(e); });
    return { id };
  }
  if (p.endsWith("/api/is")) { const j = WJOBS[q.get("id")]; if (!j) throw new Error("iş bulunamadı");
    return { durum: j.durum, mesaj: j.mesaj, yapilan: j.yapilan, toplam: j.toplam, onbellek: 0, sure: Math.round((performance.now() - j.t0) / 1000) }; }
  if (p.endsWith("/api/sonuc")) { const j = WJOBS[q.get("id")]; if (!j || !j.sonuc) throw new Error("sonuç hazır değil"); return j.sonuc; }
  throw new Error("Bu işlem internet sürümünde yok.");
}
/* ---- yer arama (Nominatim, saniyede en fazla 1 istek, oturum onbellegi) ---- */
async function wNom(path, params) {
  const key = path + JSON.stringify(params);
  if (WNOM.has(key)) return WNOM.get(key);
  const wait = 1100 - (Date.now() - W_NOM_LAST); if (wait > 0) await new Promise(r => setTimeout(r, wait));
  W_NOM_LAST = Date.now();
  let r;
  try { r = await fetch("https://nominatim.openstreetmap.org/" + path + "?" + new URLSearchParams(params)); }
  catch (e) { throw new Error("Yer arama servisine ulaşılamadı (internet bağlantısını kontrol et)."); }
  if (!r.ok) throw new Error("Yer arama servisi hata verdi (" + r.status + "). Biraz sonra tekrar dene.");
  const j = await r.json(); WNOM.set(key, j); return j;
}
function wBoxGeom(w, s, e, n) { return { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] }; }
function wPlace(it) {
  const bb = (it.boundingbox || [0, 0, 0, 0]).map(Number);
  let w = bb[2], s = bb[0], e = bb[3], n = bb[1];
  let geom = it.geojson || {};
  if (geom.type !== "Polygon" && geom.type !== "MultiPolygon") {
    const pad = .15, cx = +(it.lon != null ? it.lon : (w + e) / 2), cy = +(it.lat != null ? it.lat : (s + n) / 2);
    w = Math.min(w, cx - pad); s = Math.min(s, cy - pad); e = Math.max(e, cx + pad); n = Math.max(n, cy + pad);
    geom = wBoxGeom(w, s, e, n);
  }
  const addr = it.address || {}, name = it.name || (it.display_name || "").split(",")[0];
  return { ad: name, tam: it.display_name || name, tur: it.addresstype || it.type || "", ulke: addr.country || "",
           bbox: [w, s, e, n], lat: +(it.lat != null ? it.lat : (s + n) / 2), lon: +(it.lon != null ? it.lon : (w + e) / 2), geojson: geom };
}
async function webSearch(q) {
  const m = q.match(/^\s*(-?\d+(?:[.,]\d+)?)\s*[,; ]\s*(-?\d+(?:[.,]\d+)?)\s*$/);
  if (m) { const la = +m[1].replace(",", "."), lo = +m[2].replace(",", ".");
    if (Math.abs(la) <= 90 && Math.abs(lo) <= 180) return [{ ad: `${la.toFixed(2)}, ${lo.toFixed(2)} çevresi`, tam: `${la.toFixed(2)}° enlem, ${lo.toFixed(2)}° boylam çevresinde 1° × 1° kutu`, tur: "kutu", ulke: "",
      bbox: [lo - .5, la - .5, lo + .5, la + .5], lat: la, lon: lo, geojson: wBoxGeom(lo - .5, la - .5, lo + .5, la + .5) }]; }
  const res = await wNom("search", { q, format: "jsonv2", polygon_geojson: 1, polygon_threshold: 0.005, addressdetails: 1, limit: 8, "accept-language": "tr,en" });
  const out = res.map(wPlace);
  out.sort((a, b) => (a.geojson.type !== "Polygon" || a.geojson.coordinates[0].length > 5 ? 0 : 1) - (b.geojson.type !== "Polygon" || b.geojson.coordinates[0].length > 5 ? 0 : 1));
  return out;
}
async function webReverse(lat, lon, level) {
  const it = await wNom("reverse", { lat: lat.toFixed(4), lon: lon.toFixed(4), zoom: W_LEVEL_ZOOM[level] || 5, format: "jsonv2", polygon_geojson: 1, polygon_threshold: 0.005, addressdetails: 1, "accept-language": "tr,en" });
  if (!it || it.error) throw new Error("Bu noktada bir yerleşim sınırı bulunamadı (deniz olabilir).");
  return wPlace(it);
}
/* ---- geometri ---- */
function wRings(geom) {
  const t = geom && geom.type, c = geom && geom.coordinates, polys = t === "Polygon" ? [c] : t === "MultiPolygon" ? c : [], out = [];
  for (const poly of polys) for (const ring of poly) {
    if (!ring || ring.length < 3) continue;
    const a = ring.map(p => [+p[0], +p[1]]);
    if (a[0][0] !== a[a.length - 1][0] || a[0][1] !== a[a.length - 1][1]) a.push([a[0][0], a[0][1]]);
    out.push(a);
  }
  return out;
}
function wPointsIn(px, py, rings) {      /* cift-tek kurali; noktalar enleme gore siralanir (sunucu.py points_in ile ayni) */
  const n = px.length, order = new Uint32Array(n); for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => py[a] - py[b]);
  const ys = new Float64Array(n), xs = new Float64Array(n); for (let i = 0; i < n; i++) { ys[i] = py[order[i]]; xs[i] = px[order[i]]; }
  const par = new Uint8Array(n), lb = v => { let lo = 0, hi = n; while (lo < hi) { const m = (lo + hi) >> 1; if (ys[m] < v) lo = m + 1; else hi = m; } return lo; };
  for (const ring of rings) for (let k = 0; k < ring.length - 1; k++) {
    const x0 = ring[k][0], y0 = ring[k][1], x1 = ring[k + 1][0], y1 = ring[k + 1][1];
    if (y0 === y1) continue;
    const i0 = lb(Math.min(y0, y1)), i1 = lb(Math.max(y0, y1)), sl = (x1 - x0) / (y1 - y0);
    for (let i = i0; i < i1; i++) if (xs[i] < x0 + (ys[i] - y0) * sl) par[i] ^= 1;
  }
  const out = new Uint8Array(n); for (let i = 0; i < n; i++) out[order[i]] = par[i]; return out;
}
/* ---- karo okuma ---- */
async function wTile(ti, tj) {
  const key = ti + "_" + tj;
  if (WTILES.has(key)) return WTILES.get(key);
  if (!WEB.tileDir) { WEB.tileDir = new Map(); WEB.tiles.forEach((t, i) => WEB.tileDir.set(t, (WEB.dirs || [])[i] || 1)); }
  if (!WEB.tileDir.has(key)) { WTILES.set(key, null); return null; }
  const r = await fetch(WEB.flat ? `T${key}.dat` : `veri${WEB.tileDir.get(key)}/T${key}.dat`);
  if (!r.ok) throw new Error(`Veri parçası indirilemedi (T${key}, ${r.status}).`);
  const buf = await new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
  const dv = new DataView(buf);
  if (String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3)) !== "ATK1") throw new Error("Veri parçası bozuk: T" + key);
  const base = dv.getInt32(4, true), n = dv.getUint32(8, true), col = i => new Uint16Array(buf, 12 + i * 2 * n, n);
  const t = { ti, tj, base, n, d: col(0), c: col(1), M: col(2), V: col(3) };
  WTILES.set(key, t);
  if (WTILES.size > 60) WTILES.delete(WTILES.keys().next().value);
  return t;
}
/* ---- yontem (build_app.py / sunucu.py ile ayni) ---- */
const wDayOf = (y, m, d) => Math.round((Date.UTC(y, m, d) - W_DAY0) / 864e5);
const wYearOf = d => new Date(W_DAY0 + d * 864e5).getUTCFullYear();
const wIso = d => new Date(W_DAY0 + d * 864e5).toISOString().slice(0, 10);
function wSplit(ov) {
  const nom = ov.filter(y => y <= 2020);
  if (nom.length >= 3) { const ntr = Math.max(2, Math.round(nom.length * 2 / 3)); return [nom.slice(0, ntr), nom.slice(ntr), ov.filter(y => y > 2020)]; }
  const ntr = Math.max(1, Math.round(ov.length * 0.6)); return [ov.slice(0, ntr), ov.slice(ntr), []];
}
const wYrLabel = ys => !ys.length ? "" : ys.length === 1 ? String(ys[0]) : `${ys[0]}–${ys[ys.length - 1]}`;
function wFit(V, M, idx, minN) {
  const nb = W_EDGES.length - 1, cnt = new Float64Array(nb), sv = new Float64Array(nb), sm = new Float64Array(nb);
  for (const i of idx) { const v = V[i]; if (!(v > 0)) continue; let lo = 0, hi = nb - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (v > W_EDGES[mid + 1]) lo = mid + 1; else hi = mid; }   /* (a, b] araliklari */
    cnt[lo]++; sv[lo] += v; sm[lo] += M[i]; }
  const x = [], y = [];
  for (let b = 0; b < nb; b++) if (cnt[b] >= minN) { x.push(sv[b] / cnt[b]); y.push(Math.max(1e-4, sm[b] / cnt[b])); }
  if (x.length < 4) return null;
  for (let i = 1; i < y.length; i++) y[i] = Math.max(y[i], y[i - 1]);
  return [x, y];
}
function wCurveF(V, x, y, sHi) {
  const lx = x.map(Math.log), ly = y.map(v => Math.log(Math.max(v, 1e-6))), L = lx.length;
  const hi = sHi == null ? (ly[L - 1] - ly[L - 2]) / (lx[L - 1] - lx[L - 2]) : sHi, lo = (ly[1] - ly[0]) / (lx[1] - lx[0]);
  const out = new Float64Array(V.length);
  for (let i = 0; i < V.length; i++) { const v = V[i]; if (!(v > 0)) continue; const lv = Math.log(v); let e;
    if (lv > lx[L - 1]) e = ly[L - 1] + hi * (lv - lx[L - 1]);
    else if (lv < lx[0]) e = ly[0] + lo * (lv - lx[0]);
    else { let a = 0, b = L - 1; while (b - a > 1) { const m = (a + b) >> 1; if (lx[m] <= lv) a = m; else b = m; }
      e = lx[b] === lx[a] ? ly[a] : ly[a] + (ly[b] - ly[a]) * (lv - lx[a]) / (lx[b] - lx[a]); }
    out[i] = Math.exp(e); }
  return out;
}
function wCorr(a, b) {
  const n = a.length; let ma = 0, mb = 0; for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n;
  let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  return sab / Math.sqrt(saa * sbb);
}
function wSumBy(keys, vals, mask) { const m = new Map(); for (let i = 0; i < keys.length; i++) { if (mask && !mask[i]) continue; m.set(keys[i], (m.get(keys[i]) || 0) + vals[i]); } return m; }
function wPayload(R, name, place, note) {
  const n = R.d.length, yil = new Int32Array(n), yc = new Map();
  for (let i = 0; i < n; i++) { const d = R.d[i]; let y = yc.get(d); if (y === undefined) { y = wYearOf(d); yc.set(d, y); } yil[i] = y; }
  const D = WEB.default; let x = D.x, y = D.y, hi = D.hi, how = "Türkiye çok yıllı eğrisi (bölgeye özgü eğri için tespit az)", curve = null;
  const sV = wSumBy(yil, R.V), sM = wSumBy(yil, R.M);
  const ov = [...sV.keys()].filter(yr => sV.get(yr) > 0 && (sM.get(yr) || 0) > 0).sort((a, b) => a - b);
  let A = [], T = [], DR = [], fit = null;
  const idxWhere = f => { const o = []; for (let i = 0; i < n; i++) if (f(i)) o.push(i); return o; };
  if (ov.length >= 3) { [A, T, DR] = wSplit(ov); const s = new Set(A); fit = wFit(R.V, R.M, idxWhere(i => s.has(yil[i])), 12); }
  else { const s = new Set(ov), sub = idxWhere(i => s.has(yil[i]));
    if (sub.length) { const days = [...new Set(sub.map(i => R.d[i]))].sort((a, b) => a - b), cut = days[Math.floor(days.length * 0.6)];
      fit = wFit(R.V, R.M, sub.filter(i => R.d[i] < cut), 12); } }
  if (fit) { [x, y] = fit; hi = 1.0; how = "bölgeye özgü eğri"; curve = { x: x.map(v => +v.toFixed(4)), y: y.map(v => +v.toFixed(5)), hi: 1.0 }; }
  const f = wCurveF(R.V, x, y, hi);
  /* calibrate */
  const sF = wSumBy(yil, f), ov2 = [...sF.keys()].filter(yr => sF.get(yr) > 0 && (sM.get(yr) || 0) > 0).sort((a, b) => a - b);
  let k = D.k, calib, tr, te;
  if (!ov2.length) { calib = "bu bölgede MODIS ve VIIRS'in birlikte olduğu dönem yok; varsayılan katsayı."; }
  else {
    let labTr, labTe;
    if (ov2.length >= 3) { const [a, t] = wSplit(ov2), sa = new Set(a), st = new Set(t);
      tr = Uint8Array.from(yil, v => sa.has(v)); te = Uint8Array.from(yil, v => st.has(v)); labTr = wYrLabel(a); labTe = wYrLabel(t); }
    else { const so = new Set(ov2), sub = idxWhere(i => so.has(yil[i])), days = [...new Set(sub.map(i => R.d[i]))].sort((a, b) => a - b), cut = days[Math.floor(days.length * 0.6)];
      tr = new Uint8Array(n); te = new Uint8Array(n); for (const i of sub) { if (R.d[i] < cut) tr[i] = 1; else te[i] = 1; }
      labTr = `${wIso(days[0])} – ${wIso(cut - 1)}`; labTe = `${wIso(cut)} sonrası`; }
    let mTr = 0, fTr = 0, mTe = 0, fTe = 0, anyTe = false;
    for (let i = 0; i < n; i++) { if (tr[i]) { mTr += R.M[i]; fTr += f[i]; } if (te[i]) { anyTe = true; mTe += R.M[i]; fTe += f[i]; } }
    k = mTr / Math.max(fTr, 1e-9);
    const ratio = anyTe ? k * fTe / Math.max(mTe, 1) : null;
    calib = `eğitim ${labTr}` + (labTe ? `, test ${labTe}` : "") + (ratio ? `; test döneminde toplam isabet %${(100 * ratio).toFixed(0)}` : "");
  }
  let extra = "";
  if (T.length) {
    const st = new Set(T), mask = Uint8Array.from(yil, v => st.has(v));
    const dm = wSumBy(R.d, R.M, mask), df = wSumBy(R.d, f, mask);
    if (dm.size > 2) { const ks = [...dm.keys()]; extra = `; test günlük seyir r=${wCorr(ks.map(q => Math.log1p(dm.get(q))), ks.map(q => Math.log1p(k * df.get(q)))).toFixed(2)}`; }
    const wk = Array.from(R.d, v => Math.floor(v / 7)), wm = wSumBy(wk, R.M, mask), wf = wSumBy(wk, f, mask);
    if (wm.size > 4) { const ks = [...wm.keys()]; extra += `, haftalık r=${wCorr(ks.map(q => Math.log1p(wm.get(q))), ks.map(q => Math.log1p(k * wf.get(q)))).toFixed(2)}`; }
  }
  if (DR.length) extra += "; yörünge kayması yıllarında ham MODIS ÷ uyumlu seri: " + DR.slice().sort((a, b) => a - b).map(yr => `${yr} %${Math.round(sM.get(yr) / Math.max(k * sF.get(yr), 1e-9) * 100)}`).join(", ");
  if (!curve) curve = { x: x.map(v => +(+v).toFixed(4)), y: y.map(v => +(+v).toFixed(5)), hi };
  /* hucreler */
  const cix = new Map(), cl = [], co = [], c = new Int32Array(n);
  for (let i = 0; i < n; i++) { const key = R.lat[i] * 1e4 + R.lon[i]; let j = cix.get(key); if (j === undefined) { j = cl.length; cix.set(key, j); cl.push(+R.lat[i].toFixed(3)); co.push(+R.lon[i].toFixed(3)); } c[i] = j; }
  /* yillik belirsizlik */
  let unc = null;
  const nomOv = ov.filter(yr => yr <= 2020);
  if (nomOv.length >= 3) { const vals = nomOv.filter(yr => sM.get(yr) > 0).map(yr => Math.log(k * sF.get(yr) / sM.get(yr)));
    if (vals.length >= 3) { const mu = vals.reduce((a, b) => a + b, 0) / vals.length; unc = +Math.sqrt(vals.reduce((a, b) => a + (b - mu) ** 2, 0) / (vals.length - 1)).toFixed(4); if (!isFinite(unc)) unc = null; } }
  let nTr = 0; if (A.length) { const sa = new Set(A); for (let i = 0; i < n; i++) if (sa.has(yil[i])) nTr += R.M[i]; } else for (let i = 0; i < n; i++) nTr += R.M[i];
  const guven = unc != null && unc < 0.12 && nTr >= 3000 ? "yüksek" : unc != null && unc < 0.25 && nTr >= 800 ? "orta" : "düşük";
  const calibFull = `Kalibrasyon yıllarında ${nTr.toLocaleString("en-US").replace(/,/g, ".")} MODIS tespiti. ${how}; ${calib}${extra}`;
  return { unc, guven, key: "Q" + Math.random().toString(16).slice(2, 10), name, cell: W_CELL, k: +k.toFixed(4), calib: calibFull, note,
    cities: place.lat != null ? [{ n: place.ad || name, lat: place.lat, lon: place.lon }] : [],
    cells: { lat: cl, lon: co }, c: Array.from(c), d: Array.from(R.d), m: Array.from(R.M), v: Array.from(R.V, v => Math.round(v * 10)),
    curve, boundary: place.geojson };
}
async function webRegion(sp, job) {
  const place = sp.yer || {};
  const geom = place.geojson || (sp.bbox ? wBoxGeom(...sp.bbox) : null);
  if (!geom) throw new Error("Bölge sınırı yok.");
  place.geojson = geom;
  const rings = wRings(geom); if (!rings.length) throw new Error("Sınır geometrisi okunamadı.");
  let w = 180, s = 90, e = -180, nn = -90;
  for (const r of rings) for (const p of r) { w = Math.min(w, p[0]); e = Math.max(e, p[0]); s = Math.min(s, p[1]); nn = Math.max(nn, p[1]); }
  const Y0 = WEB.years[0], Y1 = WEB.years[WEB.years.length - 1];
  let y0 = Math.min(+sp.y0 || Y0, +sp.y1 || Y1), y1 = Math.max(+sp.y0 || Y0, +sp.y1 || Y1);
  let clipNote = "";
  if (y0 < Y0 || y1 > Y1) { clipNote = ` İnternet sürümünde veri ${Y0}–${Y1} arası; dönem buna göre kırpıldı.`; y0 = Math.max(y0, Y0); y1 = Math.min(y1, Y1); }
  if (y0 > y1) throw new Error(`İnternet sürümünde yalnızca ${Y0}–${Y1} yılları var.`);
  const tis = [], tjs = [];
  for (let a = Math.floor(s / 10); a <= Math.floor(Math.min(nn, 89.99) / 10); a++) tis.push(a);
  for (let b = Math.floor(w / 10); b <= Math.floor(Math.min(e, 179.99) / 10); b++) tjs.push(b);
  const list = []; for (const a of tis) for (const b of tjs) list.push([a, b]);
  if (list.length > W_MAXTILE) throw new Error(`Bölge internet sürümü için çok geniş (${Math.round(e - w)}° × ${Math.round(nn - s)}°). Daha küçük bir alan seç ya da bilgisayardaki tam sürümü kullan.`);
  job.toplam = list.length; job.mesaj = "veri parçaları indiriliyor";
  const tiles = [];
  for (let i = 0; i < list.length; i += 6) {
    const got = await Promise.all(list.slice(i, i + 6).map(([a, b]) => wTile(a, b).then(t => { job.yapilan++; return t; })));
    tiles.push(...got.filter(Boolean));
  }
  job.mesaj = "hesaplanıyor";
  await new Promise(r => setTimeout(r, 0));
  const d0 = wDayOf(y0, 0, 1), d1 = wDayOf(y1, 11, 31);
  const lat = [], lon = [], dd = [], MM = [], VV = [];
  for (const t of tiles) {
    for (let i = 0; i < t.n; i++) {
      const d = t.base + t.d[i]; if (d < d0 || d > d1) continue;
      const ci = (t.c[i] / 40) | 0, cj = t.c[i] - ci * 40, la = t.ti * 10 + (ci + .5) * W_CELL, lo = t.tj * 10 + (cj + .5) * W_CELL;
      if (lo < w - W_CELL || lo > e + W_CELL || la < s - W_CELL || la > nn + W_CELL) continue;
      lat.push(la); lon.push(lo); dd.push(d); MM.push(t.M[i]); VV.push(t.V[i] / 10);
    }
  }
  if (!dd.length) throw new Error("Bu bölgede ve dönemde hiç yangın tespiti yok.");
  /* hucrelerin sinir icindeki alan payi: 6 x 6 ornek nokta */
  const cmap = new Map(), cla = [], clo = [];
  for (let i = 0; i < lat.length; i++) { const key = lat[i] * 1e4 + lon[i]; if (!cmap.has(key)) { cmap.set(key, cla.length); cla.push(lat[i]); clo.push(lon[i]); } }
  const K = 6, nc = cla.length, px = new Float64Array(nc * K * K), py = new Float64Array(nc * K * K);
  for (let j = 0, q = 0; j < nc; j++) for (let gy = 0; gy < K; gy++) for (let gx = 0; gx < K; gx++, q++) {
    px[q] = clo[j] + ((gx + .5) / K * W_CELL - W_CELL / 2); py[q] = cla[j] + ((gy + .5) / K * W_CELL - W_CELL / 2); }
  const inside = wPointsIn(px, py, rings), frac = new Float64Array(nc);
  for (let j = 0; j < nc; j++) { let a = 0; for (let q = j * K * K; q < (j + 1) * K * K; q++) a += inside[q]; frac[j] = a / (K * K); }
  let keep = new Uint8Array(nc), nk = 0, rule = "alanının en az yarısı sınırın içinde kalan 0,25° hücreler";
  for (let j = 0; j < nc; j++) if (frac[j] >= .5) { keep[j] = 1; nk++; }
  if (!nk) { const mx = Math.max(...frac); if (mx > 0) for (let j = 0; j < nc; j++) if (frac[j] === mx) { keep[j] = 1; nk++; }
    rule = "bölge bir hücreden küçük; en çok örtüşen 0,25° hücre"; }
  if (!nk) throw new Error("Bu bölgede ve dönemde hiç yangın tespiti yok.");
  const R = { lat: [], lon: [], d: [], M: [], V: [] };
  for (let i = 0; i < lat.length; i++) { if (!keep[cmap.get(lat[i] * 1e4 + lon[i])]) continue; if (!(MM[i] > 0) && !(VV[i] > 0)) continue;
    R.lat.push(lat[i]); R.lon.push(lon[i]); R.d.push(dd[i]); R.M.push(MM[i]); R.V.push(VV[i]); }
  if (!R.d.length) throw new Error("Bu bölgede ve dönemde hiç yangın tespiti yok.");
  const note = `Hazır dünya verisinden tarayıcıda hesaplandı (Dünya veri seti ${Y0}–${Y1}, MODIS + VIIRS, kalıcı kaynaklar ayıklanmış). Kullanılan: ${rule}, ${nk} hücre.` + clipNote;
  const obj = wPayload(R, place.ad || "Seçilen alan", place, note);
  obj.name = place.ad || "Seçilen alan";
  obj.donem = y0 !== y1 ? `${y0}–${y1}` : String(y0);
  job.mesaj = `${R.d.length.toLocaleString("en-US").replace(/,/g, ".")} hücre-gün hazır dünya verisinden (${((performance.now() - job.t0) / 1000).toFixed(1)} sn)`;
  return obj;
}
async function api(path, opt) {
  if (WEB) return webApi(path, opt);
  const r = await fetch(path, Object.assign({ cache: "no-store" }, opt || {}));
  let j = null; try { j = await r.json(); } catch (e) {}
  if (!r.ok) throw new Error((j && j.hata) || ("sunucu hatası " + r.status));
  return j;
}
const pct = (v, sign) => (sign ? (v >= 0 ? "+" : "−") : "") + "%" + nf(Math.abs(v), 0);
async function wProbe() {
  if (!/^https?:$/.test(location.protocol)) { SERVER = false; $("woff").hidden = false; $("won").hidden = true; return false; }
  try { const j = await api("/api/durum"); SERVER = !!(j && j.ok); } catch (e) { SERVER = false; }
  if (!SERVER && !WEB) { WEB = await webProbe(); SERVER = !!WEB; }
  $("woff").hidden = SERVER; $("won").hidden = !SERVER;
  return SERVER;
}
function loadLeaflet() {
  if (window.L) return Promise.resolve();
  // once yerel kopya (internetsiz demo, python cevrimdisi_hazirla.py), yoksa CDN
  const CDN = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/";
  const srcs = /^https?:$/.test(location.protocol) ? ["/yerel/leaflet/", CDN] : [CDN];
  const tryOne = base => new Promise((ok, bad) => {
    const js = document.createElement("script"); js.src = base + "leaflet.min.js";
    js.onload = () => { const lk = document.createElement("link"); lk.rel = "stylesheet"; lk.href = base + "leaflet.min.css"; document.head.appendChild(lk); ok(); };
    js.onerror = () => { js.remove(); bad(); }; document.head.appendChild(js);
  });
  return srcs.reduce((p, b) => p.catch(() => tryOne(b)), Promise.reject())
    .catch(() => { throw new Error("Harita kütüphanesi yüklenemedi (internet yok ve yerel kopya yok: python cevrimdisi_hazirla.py)"); });
}
function wDark() { const t = document.documentElement.getAttribute("data-theme"); return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches; }
async function wMap() {
  if (WMAP) { WMAP.invalidateSize(); return; }
  try { await loadLeaflet(); } catch (e) { wMsg(e.message, true); return; }
  WMAP = L.map("wmap", { worldCopyJump: true, minZoom: 2 }).setView([39, 30], 4);
  L.tileLayer(tileUrl(), { maxZoom: 12, attribution: "&copy; OpenStreetMap katkıcıları" }).addTo(WMAP);
  if (wDark()) $("wmap").classList.add("dark");
  let b0 = null, rect = null, skip = false;
  WMAP.on("mousedown", e => { if (WLEVEL !== "kutu") return; b0 = e.latlng; WMAP.dragging.disable();
    rect = L.rectangle([b0, b0], { color: css("--harm"), weight: 2, dashArray: "5 3", fillOpacity: .06 }).addTo(WMAP); });
  WMAP.on("mousemove", e => { if (b0 && rect) rect.setBounds([b0, e.latlng]); });
  WMAP.on("mouseup", e => { if (!b0) return; const bd = L.latLngBounds(b0, e.latlng); b0 = null; WMAP.dragging.enable(); rect.remove(); rect = null; skip = true;
    if (Math.abs(bd.getEast() - bd.getWest()) < .05 || Math.abs(bd.getNorth() - bd.getSouth()) < .05) { wMsg("Kutu çok küçük; sürükleyerek daha geniş bir alan çiz.", true); return; }
    const w = +bd.getWest().toFixed(3), s = +bd.getSouth().toFixed(3), e2 = +bd.getEast().toFixed(3), n = +bd.getNorth().toFixed(3);
    wSelect({ ad: "Çizilen kutu", tam: `${nf(s, 2)}…${nf(n, 2)}° enlem · ${nf(w, 2)}…${nf(e2, 2)}° boylam`, tur: "kutu", bbox: [w, s, e2, n],
      lat: (s + n) / 2, lon: (w + e2) / 2, geojson: { type: "Polygon", coordinates: [[[w, s], [e2, s], [e2, n], [w, n], [w, s]]] } }); });
  WMAP.on("click", async e => {
    if (skip) { skip = false; return; }
    if (WLEVEL === "kutu" || WBUSY) return;
    wMsg(`${{ ulke: "ülke", il: "il", ilce: "ilçe" }[WLEVEL]} sınırı aranıyor…`);
    try { wSelect(await api(`/api/ters?lat=${e.latlng.lat}&lon=${e.latlng.lng}&duzey=${WLEVEL}`)); wMsg(""); } catch (er) { wMsg(er.message, true); }
  });
}
const TUR = { country: "ülke", state: "il / eyalet", province: "il", county: "ilçe", city: "şehir", town: "kasaba", village: "köy", municipality: "belediye", region: "bölge", administrative: "idari bölge", district: "ilçe", suburb: "mahalle", kutu: "kutu", dosya: "yüklenen sınır" };
function wSelect(p) {
  WSELP = p;
  if (WMAP) { if (WLAYER) WLAYER.remove();
    WLAYER = L.geoJSON(p.geojson, { style: { color: css("--harm"), weight: 2, fillColor: css("--harm"), fillOpacity: .07 } }).addTo(WMAP);
    try { WMAP.fitBounds(WLAYER.getBounds(), { padding: [24, 24], maxZoom: 9 }); } catch (e) {} }
  const [w, s, e, n] = p.bbox || geoBBox(p.geojson);
  $("wsel").innerHTML = `<b>${esc(p.ad)}</b>${esc(p.tam || "")} (${TUR[p.tur] || esc(p.tur || "")}, alan ${nf(e - w, 1)}° × ${nf(n - s, 1)}°)`;
  $("wgo").disabled = WBUSY; wEstimate();
}
function esc(t) { return String(t == null ? "" : t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]); }
function wEstimate() {
  const a = Math.min(+$("wy0").value, +$("wy1").value), b2 = Math.max(+$("wy0").value, +$("wy1").value), all = $("wall").checked;
  let n = 0; for (let y = a; y <= b2; y++) { n += 73; if (y >= 2012) n += 73; if (all && y >= 2018) n += 73; if (all && y >= 2023) n += 73; }
  const big = WSELP ? (() => { const [w, s, e, nn] = WSELP.bbox || geoBBox(WSELP.geojson); return (e - w) * (nn - s); })() : 0;
  const ds = WSELP && (() => { const [w, s, e, nn] = WSELP.bbox || geoBBox(WSELP.geojson);
    return WSETS.filter(d => d.kutu[0] <= w + 1e-6 && d.kutu[1] <= s + 1e-6 && d.kutu[2] >= e - 1e-6 && d.kutu[3] >= nn - 1e-6 && d.y0 <= a && d.y1 >= b2)
      .sort((p, q) => (p.kutu[2] - p.kutu[0]) * (p.kutu[3] - p.kutu[1]) - (q.kutu[2] - q.kutu[0]) * (q.kutu[3] - q.kutu[1]))[0]; })();
  if (!ds && WSELP) { const wd = WSETS.find(d => d.key === "DUNYA");
    if (wd && (a < wd.y0 || b2 > wd.y1)) { if (+$("wy0").value < wd.y0) $("wy0").value = wd.y0; if (+$("wy1").value > wd.y1) $("wy1").value = wd.y1;
      $("west").textContent = `Dünya veri seti ${wd.y0}–${wd.y1} yıllarını kapsıyor; yıl aralığı buna göre ayarlandı. Hesap anında gelir.`; return; } }
  if (ds) { $("west").textContent = ds.key === "DUNYA" ? `Dünya veri setinden (${ds.y0}–${ds.y1}): hesap anında gelir.` : `Bu bölge "${ds.ad}" hazır veri setinin içinde (${ds.y0}–${ds.y1}): hesap anında gelir.`; return; }
  $("west").textContent = `Bu bölge hazır veri setlerinin dışında: FIRMS'ten canlı indirilecek, ≈ ${nf(n)} istek (${a}–${b2}). Daha önce indirilmiş pencereler anında gelir; yeni bir bölgede yıl başına yaklaşık ${all ? "1–2" : "½–1"} dakika.` +
    (big > 400 ? " Büyük alan: FIRMS kotası nedeniyle bekleme olabilir, önce birkaç yılla dene." : "") +
    (a > 2017 || b2 < 2014 ? " Not: kalibrasyon için 2012–2020 arasından en az 3 yıl seçmek en iyisi." : "");
}
async function wSearch(q) {
  const ol = $("wres"); ol.innerHTML = ""; wMsg("aranıyor…");
  try {
    const res = await api(`/api/ara?q=${encodeURIComponent(q)}`);
    if (!res.length) { wMsg(`"${q}" için sonuç yok. Başka yazımla dene (ör. "Mugla, Turkey").`, true); return; }
    wMsg(`${res.length} sonuç. Birini seç.`);
    res.forEach((p, i) => { const li = document.createElement("li"), bt = document.createElement("button"); bt.type = "button";
      bt.innerHTML = `<b>${esc(p.ad)}</b><em>${TUR[p.tur] || esc(p.tur)}</em><small>${esc(p.tam)}</small>`;
      bt.addEventListener("click", () => { [...ol.querySelectorAll("button")].forEach(x => x.removeAttribute("aria-current")); bt.setAttribute("aria-current", "true"); wSelect(p); });
      li.appendChild(bt); ol.appendChild(li); if (i === 0) { bt.setAttribute("aria-current", "true"); wSelect(p); } });
  } catch (e) { wMsg(e.message, true); }
}
async function wRun() {
  if (!WSELP || WBUSY) return;
  WBUSY = true; $("wgo").disabled = true; const bar = $("wprog"), fill = bar.querySelector("i"); bar.hidden = false; fill.style.width = "0"; wMsg("hesaplanıyor…");
  const y0 = +$("wy0").value, y1 = +$("wy1").value;
  try {
    const { id } = await api("/api/bolge", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yer: WSELP, y0: Math.min(y0, y1), y1: Math.max(y0, y1), tum_viirs: $("wall").checked }) });
    let st;
    for (;;) { await new Promise(r => setTimeout(r, 900)); st = await api(`/api/is?id=${id}`);
      fill.style.width = st.toplam ? (100 * st.yapilan / st.toplam).toFixed(1) + "%" : "4%";
      wMsg(`${st.mesaj} · ${st.sure} sn`);
      if (st.durum === "bitti" || st.durum === "hata") break; }
    if (st.durum === "hata") throw new Error(st.mesaj);
    fill.style.width = "100%";
    const obj = await api(`/api/sonuc?id=${id}`);
    obj.share = shareOf(WSELP, y0, y1);
    REGS[obj.key] = obj;
    showOut(true);
    cmpOptions();
    if (WSELP && WSELP.tur !== "indirilen bölge") histPut(WSELP);
    const sel = $("exreg"), o = new Option(`${obj.name} · ${obj.donem || ""}`, obj.key); sel.add(o, 0); sel.value = obj.key;
    await useRegion(obj.key);
    wMsg(`${st.mesaj}. Sonuçlar aşağıda; bölge listesine de eklendi.`);
    document.getElementById("panel-region").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  } catch (e) { wMsg(e.message, true); }
  finally { WBUSY = false; $("wgo").disabled = !WSELP; setTimeout(() => { bar.hidden = true; }, 1200); }
}
function showOut(on) {
  document.getElementById("exwait").hidden = on;
  document.querySelector(".work").hidden = !on; document.querySelector(".results").hidden = !on;
  document.getElementById("panel-region").hidden = !on;
}
let WSETS = [];
async function wSets(reload) {
  try { WSETS = await api("/api/veri_setleri" + (reload ? "?yenile=1" : "")); } catch (e) { WSETS = []; }
  const box = $("wchips"); box.innerHTML = "";
  $("wsets").hidden = !WSETS.length;

  $("wsetlab").textContent = `İndirilen bölgeler (${WSETS.filter(d => d.key !== "DUNYA").length})` + (WSETS.some(d => d.key === "DUNYA") ? " ve tüm dünya" : "");
  WSETS.filter(d => d.key !== "DUNYA").forEach(d => { const bt = document.createElement("button"); bt.type = "button"; bt.textContent = d.ad; bt.title = d.olay + ` (${d.y0}–${d.y1})`;
    bt.addEventListener("click", () => { [...box.children].forEach(x => x.removeAttribute("aria-current")); bt.setAttribute("aria-current", "true");
      const [w, s, e, n] = d.kutu;
      $("wy0").value = Math.max(d.y0, +$("wy0").value < d.y0 ? d.y0 : +$("wy0").value); $("wy1").value = d.y1;
      wSelect({ ad: d.ad, tam: d.olay, tur: "indirilen bölge", bbox: d.kutu, lat: (s + n) / 2, lon: (w + e) / 2,
        geojson: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } }); });
    box.appendChild(bt); });
  wEstimate();
}
/* son aramalar: bu tarayicida saklanir (en fazla 8; cok buyuk sinirlar saklanmaz) */
const HKEY = "atesTakvimi.sonAramalar";
function histGet() { try { return JSON.parse(localStorage.getItem(HKEY) || "[]"); } catch (e) { return []; } }
function histPut(p) {
  try { let s = JSON.stringify(p); if (s.length > 250000) return;
    const l = histGet().filter(x => !(x.ad === p.ad && x.tam === p.tam)); l.unshift(p);
    localStorage.setItem(HKEY, JSON.stringify(l.slice(0, 8))); } catch (e) {}
  wHist();
}
function wHist() {
  const l = histGet(), box = $("whchips"); box.innerHTML = ""; $("whist").hidden = !l.length;
  l.forEach(p => { const bt = document.createElement("button"); bt.type = "button"; bt.textContent = p.ad; bt.title = p.tam || "";
    bt.addEventListener("click", () => wSelect(p)); box.appendChild(bt); });
}
function wInit() {
  const yNow = new Date().getUTCFullYear(), s0 = $("wy0"), s1 = $("wy1");
  for (let y = 2003; y <= yNow; y++) { s0.add(new Option(y, y)); s1.add(new Option(y, y)); }
  s0.value = 2016; s1.value = yNow;
  ["wy0", "wy1", "wall"].forEach(id => $(id).addEventListener("change", wEstimate));
  $("wform").addEventListener("submit", e => { e.preventDefault(); const q = $("wq").value.trim(); if (q) wSearch(q); });
  document.querySelectorAll(".seg [data-lv]").forEach(bt => bt.addEventListener("click", () => {
    WLEVEL = bt.dataset.lv; document.querySelectorAll(".seg [data-lv]").forEach(x => x.setAttribute("aria-pressed", x === bt));
    $("wmap").classList.toggle("boxing", WLEVEL === "kutu"); }));
  $("wgo").addEventListener("click", wRun);
  $("wsetre").addEventListener("click", () => wSets(true));
  $("whistclr").addEventListener("click", () => { try { localStorage.removeItem(HKEY); } catch (e) {} wHist(); });
  wHist();
  wEstimate();
}

/* ================= KURE (ana sayfa) =================
   Dunya yangin haritasinin 3B kuresi: kutuphane yok, 2B canvas uzerinde ortografik izdusum.
   Suruklenir, kendi kendine doner; tiklanan yerin takvimi Hesapla'da acilir. Uc gorunum: toplam, ay ay, degisim. */
const GL = { on: true, lon: 25, lat: 18, mode: "toplam", mf: 0, play: false, spin: true, pick: null, raf: 0, last: 0, idle: 0, drag: null, ready: false, q: 1, cost: 0, NB: 24, tracks: true, auto: true, t: Date.now() / 864e5, yi: null, daySec: 60 };
try { GL.tracks = localStorage.getItem("ates_izler") !== "0"; } catch (e) {}
try { GL.on = localStorage.getItem("ates_hero") !== "duz"; } catch (e) {}
const RAD = Math.PI / 180;
/* 0,25 derecelik ince izgara (Hesapla haritasiyla ayni hucre). Veri zlib + base64 olarak gomulu; acilista bir kez acilir.
   Acilana kadar (ya da tarayici acamazsa) 1 derecelik izgara kullanilir. */
const HI = { v: null, m: null, r: null, busy: false };
async function hiLoad() {
  const h = D.hero && D.hero.hi; if (!h || HI.busy || HI.v || typeof DecompressionStream === "undefined") return; HI.busy = true;
  const dec = async b64 => { if (!b64) return null; const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
    return new Uint8Array(await new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer()); };
  try { HI.v = await dec(h.v); HI.m = await dec(h.m); HI.r = await dec(h.r); HI.w = h.w; HI.h = h.h; } catch (e) { HI.v = HI.m = HI.r = null; }
  HI.busy = false; GL.grid = null; GL.asin = null;
  if (HI.v && PAGE === "ana") { if (GL.on) globeKick(); else hero(); }
}
function globeInit() {
  if (GL.ready) return; GL.ready = true;
  GL.lon = 180 - 360 * (GL.t - Math.floor(GL.t)) + 8;                 /* acilista gunduz tarafi (ogle meridyeni) karsida */
  const c = $("globe");
  const setView = on => { GL.on = on; try { localStorage.setItem("ates_hero", on ? "kure" : "duz"); } catch (e) {} render(); };
  $("hv-kure").addEventListener("click", () => setView(true));
  $("hv-duz").addEventListener("click", () => setView(false));
  document.querySelectorAll("#gmodes [data-gm]").forEach(bt => bt.addEventListener("click", () => { GL.mode = bt.dataset.gm; GL.play = GL.mode === "ay" && !matchMedia("(prefers-reduced-motion: reduce)").matches; globeUI(); globeKick(); }));
  $("gplay").addEventListener("click", () => { GL.play = !GL.play; globeUI(); globeKick(); });
  $("gtrk").addEventListener("click", () => { GL.tracks = !GL.tracks; try { localStorage.setItem("ates_izler", GL.tracks ? "1" : "0"); } catch (e) {} globeUI(); globeKick(); });
  $("gpp").addEventListener("click", () => { GL.auto = !GL.auto; GL.spin = GL.auto; if (GL.auto) GL.pick = null; globeUI(); globeKick(); });
  $("gyr").addEventListener("input", e => { GL.yi = +e.target.value; globeUI(); globeKick(); });
  $("gmon").addEventListener("input", e => { GL.play = false; GL.mf = +e.target.value; globeUI(); globeKick(); });
  $("ggo").addEventListener("click", () => { if (GL.pick) globeJump(GL.pick[0], GL.pick[1]); });
  c.addEventListener("pointerdown", e => { TOUR.fly = null; c.setPointerCapture(e.pointerId); GL.drag = { x: e.clientX, y: e.clientY, lon: GL.lon, lat: GL.lat, moved: 0 }; GL.spin = false; });
  c.addEventListener("pointermove", e => { const d = GL.drag; if (!d) return;
    const dx = e.clientX - d.x, dy = e.clientY - d.y, k = 1 / (GL.R * RAD); d.moved = Math.max(d.moved, Math.abs(dx) + Math.abs(dy));
    GL.lon = d.lon - dx * k; GL.lat = Math.max(-80, Math.min(80, d.lat + dy * k)); globeKick(); });
  const up = e => { const d = GL.drag; GL.drag = null; if (!d) return; GL.idle = performance.now();
    if (d.moved < 5 && e.type === "pointerup") { const r = c.getBoundingClientRect(), ll = globeInv(e.clientX - r.left, e.clientY - r.top);
      if (ll) { GL.pick = ll; globeUI(); } }
    globeKick(); };
  c.addEventListener("pointerup", up); c.addEventListener("pointercancel", up);
  c.addEventListener("keydown", e => { const st = { ArrowLeft: [-10, 0], ArrowRight: [10, 0], ArrowUp: [0, 10], ArrowDown: [0, -10] }[e.key];
    if (st) { e.preventDefault(); GL.spin = false; GL.idle = performance.now(); GL.lon += st[0]; GL.lat = Math.max(-80, Math.min(80, GL.lat + st[1])); globeKick(); }
    else if (e.key === "Enter") { GL.pick = [GL.lat, ((GL.lon + 540) % 360) - 180]; globeUI(); globeKick(); } });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) globeKick(); });
  tourInit();
}
function globeInv(sx, sy) {           /* ekran -> [enlem, boylam] (kurenin disindaysa null) */
  const x = (sx - GL.cx) / GL.R, y = -(sy - GL.cy) / GL.R, q = 1 - x * x - y * y; if (q < 0) return null;
  const z = Math.sqrt(q), sT = Math.sin(GL.lat * RAD), cT = Math.cos(GL.lat * RAD), sL = Math.sin(GL.lon * RAD), cL = Math.cos(GL.lon * RAD);
  const Y = y * cT + z * sT, z1 = -y * sT + z * cT, X = x * cL + z1 * sL, Z = -x * sL + z1 * cL;
  return [Math.asin(Math.max(-1, Math.min(1, Y))) / RAD, Math.atan2(X, Z) / RAD];
}
function hiBins() {                   /* 0,25 derecelik izgara: 1440 x 720, her hucrenin renk kutusu (255: bos) */
  const H = D.hero, NB = GL.NB, W = HI.w, N = W * HI.h, q2b = (q, lo) => Math.round((lo + (1 - lo) * (q - 1) / 254) * (NB - 1));
  const key = GL.mode + (GL.mode === "ay" ? GL.mf.toFixed(2) : "");
  if (GL.grid && GL.grid.length === N && GL.gkey === key) return GL.grid;
  if (!GL.grid || GL.grid.length !== N) GL.grid = new Uint8Array(N);
  GL.gkey = key;
  const g = GL.grid, V = HI.v;
  if (GL.mode === "ay" && HI.m) {
    const M = HI.m, m0 = Math.floor(GL.mf) % 12, m1 = (m0 + 1) % 12, f = GL.mf - Math.floor(GL.mf), o0 = m0 * N, o1 = m1 * N;
    for (let i = 0; i < N; i++) { const q = M[o0 + i] * (1 - f) + M[o1 + i] * f; g[i] = q < 1 ? 255 : q2b(q, .1); }
  } else if (GL.mode === "son" && HI.r) {
    const R = HI.r; for (let i = 0; i < N; i++) g[i] = R[i] ? q2b(R[i], .15) : 255;
  } else if (GL.mode === "degisim" && H.tr) {
    /* egilim 1 derecelik hucrede hesaplanir (0,25 derecede gurultu olur); yangin olan ince hucreler o rengi alir */
    if (!GL.tr1) { GL.tr1 = new Int16Array(180 * 360).fill(-999); for (let k = 0; k < H.v.length; k++) if (H.tr[k] != null) GL.tr1[(Math.max(-90, Math.min(89, H.lat[k])) + 90) * 360 + ((H.lon[k] + 540) % 360)] = H.tr[k]; }
    const T1 = GL.tr1;
    for (let i = 0; i < N; i++) { if (!V[i]) { g[i] = 255; continue; } const r = (i / W) | 0, c = i - r * W, t = T1[(r >> 2) * 360 + (c >> 2)];
      g[i] = t === -999 ? 255 : Math.round((Math.max(-100, Math.min(100, t)) + 100) / 200 * (NB - 1)); }
  } else { for (let i = 0; i < N; i++) g[i] = V[i] ? q2b(V[i], .12) : 255; }
  return g;
}
function globeBins() {                /* 1 derecelik izgara: her hucrenin renk kutusu (255: bos) */
  if (HI.v) return hiBins();
  const H = D.hero, n = H.v.length, NB = GL.NB, LA = H.lat, LO = H.lon;
  if (GL.grid && GL.grid.length !== 180 * 360) GL.grid = null;
  if (!GL.grid) { GL.grid = new Uint8Array(180 * 360).fill(255); GL.gi = new Int32Array(n);
    for (let k = 0; k < n; k++) GL.gi[k] = (Math.max(-90, Math.min(89, LA[k])) + 90) * 360 + ((LO[k] + 540) % 360); }
  const g = GL.grid, gi = GL.gi;
  if (GL.mode === "ay" && H.m) {
    if (!GL.mmax) { let mx = 0; for (let i = 0; i < H.m.length; i++) if (H.m[i] > mx) mx = H.m[i]; GL.mmax = Math.log1p(mx); }
    const m0 = Math.floor(GL.mf) % 12, m1 = (m0 + 1) % 12, f = GL.mf - Math.floor(GL.mf), mx = GL.mmax;
    for (let k = 0; k < n; k++) { const v = H.m[k * 12 + m0] * (1 - f) + H.m[k * 12 + m1] * f;
      g[gi[k]] = v < 1.5 ? 255 : Math.round((.1 + .9 * Math.log1p(v) / mx) * (NB - 1)); }
  } else if (GL.mode === "son" && H.rc) {
    if (!GL.rmax) { let mx = 0; for (let k = 0; k < n; k++) if (H.rc.v[k] > mx) mx = H.rc.v[k]; GL.rmax = Math.log1p(mx); }
    for (let k = 0; k < n; k++) { const v = H.rc.v[k]; g[gi[k]] = v < 1 ? 255 : Math.round((.15 + .85 * Math.log1p(v) / GL.rmax) * (NB - 1)); }
  } else if (GL.mode === "degisim" && H.tr) {
    for (let k = 0; k < n; k++) { const t = H.tr[k]; g[gi[k]] = t == null ? 255 : Math.round((Math.max(-100, Math.min(100, t)) + 100) / 200 * (NB - 1)); }
  } else {
    if (!GL.vmax) { let mx = 0; for (let k = 0; k < n; k++) if (H.v[k] > mx) mx = H.v[k]; GL.vmax = Math.log1p(mx); }
    for (let k = 0; k < n; k++) g[gi[k]] = Math.round((.12 + .88 * Math.log1p(H.v[k]) / GL.vmax) * (NB - 1));
  }
  return g;
}
const gcss = n => getComputedStyle($("globe")).getPropertyValue(n).trim();   /* kure hep "uzay" renkleriyle cizilir (.hero.space) */
function globeLUT() {                 /* renk kutulari -> piksel degeri (ABGR): gunduz, alacakaranlik, gece */
  const NB = GL.NB, day = new Uint32Array(NB), dusk = new Uint32Array(NB), night = new Uint32Array(NB), bg = hex(gcss("--ground"));
  const pk = c => (255 << 24 | c[2] << 16 | c[1] << 8 | c[0]) >>> 0, dim = (c, f) => c.map((v, k) => Math.round(v + (bg[k] - v) * f));
  const put = (i, c) => { day[i] = pk(c); dusk[i] = pk(dim(c, .22)); night[i] = pk(dim(c, .45)); };
  if (GL.mode === "degisim" && D.hero.tr) {
    const lo = hex(gcss("--viirs")), hi = hex(gcss("--harm")), mid = hex(gcss("--ground3"));
    for (let i = 0; i < NB; i++) { const t = i / (NB - 1) * 2 - 1, a = t < 0 ? lo : hi, f = Math.pow(Math.abs(t), .7); put(i, mid.map((v, k) => Math.round(v + (a[k] - v) * f))); }
  } else { const st = ["--t0", "--t1", "--t2", "--t3", "--t4"].map(v => hex(gcss(v)));
    for (let i = 0; i < NB; i++) { const x = i / (NB - 1) * (st.length - 1), j = Math.min(Math.floor(x), st.length - 2), f = x - j;
      put(i, st[j].map((v, k) => Math.round(v + (st[j + 1][k] - v) * f))); } }
  return { day, dusk, night };
}
function globeRaster(R, dpr, sT, cT, sL, cL, sun) {
  /* Kure yuzeyi piksel piksel boyanir: her piksel icin ters izdusumle enlem-boylam bulunur, 1 derecelik izgaradan renk okunur.
     Maliyet hucre sayisindan bagimsiz; hareket ederken cozunurluk dusurulur (GL.q). sun: gunes yonu (varsa gece tarafi golgelenir). */
  const S = Math.max(96, Math.min(HI.v ? 1100 : 820, Math.round(2 * R * dpr * GL.q)));
  if (!GL.px || GL.px.S !== S) {
    const cv = document.createElement("canvas"); cv.width = cv.height = S; const c2 = cv.getContext("2d"), img = c2.createImageData(S, S);
    const xs = [], ys = [], zs = [], pos = [], h = S / 2;
    for (let py = 0; py < S; py++) for (let px = 0; px < S; px++) { const x = (px + .5 - h) / h, y = -(py + .5 - h) / h, q = 1 - x * x - y * y;
      if (q > 0) { xs.push(x); ys.push(y); zs.push(Math.sqrt(q)); pos.push(py * S + px); } }
    GL.px = { S, cv, c2, img, buf: new Uint32Array(img.data.buffer), xs: Float32Array.from(xs), ys: Float32Array.from(ys), zs: Float32Array.from(zs), pos: Int32Array.from(pos) };
  }
  const g = globeBins(), GW = HI.v ? HI.w : 360, GH = HI.v ? HI.h : 180, cpd = GW / 360;   /* hucre/derece: 4 (ince) ya da 1 */
  if (!GL.asin || GL.asinH !== GH) { const L = 16384; GL.asinH = GH; GL.asin = new Uint16Array(L + 1);
    for (let i = 0; i <= L; i++) GL.asin[i] = Math.min(GH - 1, Math.max(0, Math.floor((Math.asin(Math.max(-1, Math.min(1, i / (L / 2) - 1))) / RAD + 90) * cpd))); }
  const { cv, c2, img, buf, xs, ys, zs, pos } = GL.px, lut = globeLUT(), AS = GL.asin, n = pos.length, K = 180 / Math.PI * cpd, HL = 8192, LM = GW - 1, L180 = 180 * cpd;
  const LD = lut.day, LK = lut.dusk, LN = lut.night, NIGHT = (150 << 24 | 12 << 16 | 6 << 8 | 3) >>> 0, DUSK = (70 << 24 | 12 << 16 | 6 << 8 | 3) >>> 0;
  const s0 = sun ? sun[0] : 0, s1 = sun ? sun[1] : 0, s2 = sun ? sun[2] : 1, useSun = !!sun;
  for (let i = 0; i < n; i++) {
    const x = xs[i], y = ys[i], z = zs[i], Y = y * cT + z * sT, z1 = z * cT - y * sT, X = x * cL + z1 * sL, Z = z1 * cL - x * sL;
    let lo = (Math.atan2(X, Z) * K + L180) | 0; if (lo > LM) lo = LM;
    const b = g[AS[((Y + 1) * HL) | 0] * GW + lo];
    if (useSun) { const d = X * s0 + Y * s1 + Z * s2;
      buf[pos[i]] = d > .07 ? (b === 255 ? 0 : LD[b]) : d > -.07 ? (b === 255 ? DUSK : LK[b]) : (b === 255 ? NIGHT : LN[b]); }
    else buf[pos[i]] = b === 255 ? 0 : LD[b];
  }
  c2.putImageData(img, 0, 0);
  return cv;
}
/* ---- uydular: gunes-esz zamanli yorungeler ----
   Yorunge duzlemi Gunes'e gore sabittir; Dunya altinda gunde bir kez doner. Duzlemin yeri, uydunun Turkiye enleminden (39 K)
   gunduz gecisinin olculen yerel saatinden hesaplanir (D.drift24: FIRMS tespitlerinden yillik medyan). Yukseklikler gercek olcekte. */
const SATS = [
  { nm: "Terra", cv: "--terra", inc: 98.2, per: 98.9, alt: 705, asc: false, key: "terra", u0: 1.0, ghost: true },
  { nm: "Aqua", cv: "--aqua", inc: 98.2, per: 98.9, alt: 705, asc: true, key: "aqua", u0: 2.6, ghost: true },
  { nm: "S-NPP", cv: "--snpp", inc: 98.7, per: 101.4, alt: 824, asc: true, key: "snpp", u0: 0, ring: "VIIRS" },
  { nm: "NOAA-20", cv: "--snpp", inc: 98.7, per: 101.4, alt: 824, asc: true, key: "noaa20", u0: Math.PI, ring: "VIIRS" }];
function orbPt(u, inc, lam0) { const si = Math.sin(inc * RAD), ci = Math.cos(inc * RAD), su = Math.sin(u), cu = Math.cos(u); return [Math.asin(si * su) / RAD, lam0 + Math.atan2(su * ci, cu) / RAD]; }
function globeSky() {
  const dr = D.drift24 || D.drift; if (!dr || !dr.years) return null;
  const yi = Math.max(0, Math.min(dr.years.length - 1, GL.yi == null ? dr.years.length - 1 : GL.yi)), fr = GL.t - Math.floor(GL.t), ls = 180 - 360 * fr;
  if (GL.decl == null) { const d = new Date(), N = Math.floor((d - new Date(Date.UTC(d.getUTCFullYear(), 0, 0))) / 864e5); GL.decl = -23.44 * Math.cos(2 * Math.PI * (N + 10) / 365); }
  const cd = Math.cos(GL.decl * RAD), sun = [cd * Math.sin(ls * RAD), Math.sin(GL.decl * RAD), cd * Math.cos(ls * RAD)];
  const first = a => { const i = (a || []).findIndex(v => v != null); return i < 0 ? null : a[i]; };
  const plane = (s, lt) => { const us = Math.asin(Math.sin(39 * RAD) / Math.sin(s.inc * RAD)), u = s.asc ? us : Math.PI - us;      /* 39 K gunduz gecisi */
    return { lam0: ls + (lt - 12) * 15 - orbPt(u, s.inc, 0)[1], ulab: u }; };
  const sats = [], rings = {};
  for (const s of SATS) { const lt = (dr[s.key] || [])[yi]; if (lt == null) continue;
    const pl = plane(s, lt), r = 1 + s.alt / 6371, rk = s.ring || s.nm;
    if (!rings[rk]) { const pts = []; for (let k = 0; k <= 120; k++) pts.push(orbPt(k / 120 * 2 * Math.PI, s.inc, pl.lam0));
      let gh = null; const l0 = first(dr[s.key]);
      if (s.ghost && l0 != null && Math.abs(l0 - lt) > .25) { const g0 = plane(s, l0); gh = []; for (let k = 0; k <= 120; k++) gh.push(orbPt(k / 120 * 2 * Math.PI, s.inc, g0.lam0)); }
      rings[rk] = { nm: rk, cv: s.cv, r, pts, ghost: gh, lab: orbPt(pl.ulab, s.inc, pl.lam0), lt }; }
    sats.push({ nm: s.nm, cv: s.cv, r, at: orbPt(s.u0 + 2 * Math.PI * GL.t * 1440 / s.per, s.inc, pl.lam0) }); }
  return { sun, ls, year: dr.years[yi], rings: Object.values(rings), sats, firstYear: dr.years[0] };
}
function globeStars(w, h, dpr) {
  if (GL.stars && GL.stars.w === w && GL.stars.h === h) return GL.stars.cv;
  const cv = document.createElement("canvas"); cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); const c = cv.getContext("2d"); c.scale(dpr, dpr);
  let s = 20261114; const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  const n = Math.round(w * h / 1500), tint = ["#ffffff", "#dfe7ff", "#fff1d8"];
  for (let i = 0; i < n; i++) { const x = rnd() * w, y = rnd() * h, m = rnd(), r = m > .96 ? 1.7 : m > .75 ? 1.1 : .75;
    c.globalAlpha = .3 + rnd() * .65; c.fillStyle = tint[(rnd() * 3) | 0]; c.fillRect(x, y, r, r); }
  GL.stars = { w, h, cv }; return cv;
}
function globeDraw() {
  const H = D.hero, c = $("globe"); if (!H || c.hidden) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1), w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const ctx = c.getContext("2d"), R = GL.R = Math.min(h * .41, w * .41), cx = GL.cx = w / 2, cy = GL.cy = h / 2;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  ctx.drawImage(globeStars(w, h, dpr), 0, 0, w, h);
  const sT = Math.sin(GL.lat * RAD), cT = Math.cos(GL.lat * RAD), sL = Math.sin(GL.lon * RAD), cL = Math.cos(GL.lon * RAD);
  /* (enlem, boylam, yaricap) -> ekran; z: on yuz > 0 */
  const P = [0, 0, 0], proj = (la, lo, r) => { const k = r || 1, cp = Math.cos(la * RAD), X = cp * Math.sin(lo * RAD), Y = Math.sin(la * RAD), Z = cp * Math.cos(lo * RAD);
    const x1 = X * cL - Z * sL, z1 = X * sL + Z * cL; P[0] = cx + R * k * x1; P[1] = cy - R * k * (Y * cT - z1 * sT); P[2] = k * (Y * sT + z1 * cT); };
  const sky = GL.tracks ? globeSky() : null, C = n => gcss(n);
  /* yorunge halkalari: on ve arka yarim ayri yollar (arka yarim kureden once cizilir, kure onu orter) */
  const split = (pts, r) => { const f = new Path2D(), b = new Path2D(); let pf = false, pb = false;
    for (const [la, lo] of pts) { proj(la, lo, r); if (P[2] >= 0) { pf ? f.lineTo(P[0], P[1]) : f.moveTo(P[0], P[1]); pf = true; pb = false; } else { pb ? b.lineTo(P[0], P[1]) : b.moveTo(P[0], P[1]); pb = true; pf = false; } }
    return [f, b]; };
  const ringPaths = sky ? sky.rings.map(rg => ({ rg, main: split(rg.pts, rg.r), ghost: rg.ghost ? split(rg.ghost, rg.r) : null })) : [];
  const sat = (s, front) => { proj(s.at[0], s.at[1], s.r); if ((P[2] >= 0) !== front) return; const x = P[0], y = P[1], k = front ? 5.5 : 4;
    ctx.beginPath(); ctx.moveTo(x, y - k); ctx.lineTo(x + k, y); ctx.lineTo(x, y + k); ctx.lineTo(x - k, y); ctx.closePath();
    ctx.globalAlpha = front ? 1 : .55; ctx.fillStyle = C(s.cv); ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = "#080b14"; ctx.stroke(); ctx.globalAlpha = 1; };
  ctx.lineJoin = ctx.lineCap = "round";
  ringPaths.forEach(({ rg, main, ghost }) => { ctx.strokeStyle = C(rg.cv);
    if (ghost) { ctx.setLineDash([5, 6]); ctx.globalAlpha = .3; ctx.lineWidth = 1.1; ctx.stroke(ghost[1]); ctx.setLineDash([]); }
    ctx.globalAlpha = .4; ctx.lineWidth = 1.2; ctx.stroke(main[1]); ctx.globalAlpha = 1; });
  if (sky) sky.sats.forEach(s => sat(s, false));
  /* kure */
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.fillStyle = C("--ground"); ctx.fill();
  ctx.strokeStyle = C("--line"); ctx.lineWidth = .7; ctx.beginPath();
  const seg = (pts) => { let pen = false; for (const [la, lo] of pts) { proj(la, lo); if (P[2] > 0) { pen ? ctx.lineTo(P[0], P[1]) : ctx.moveTo(P[0], P[1]); pen = true; } else pen = false; } };
  for (let lo = -180; lo < 180; lo += 30) { const pts = []; for (let la = -90; la <= 90; la += 5) pts.push([la, lo]); seg(pts); }
  for (let la = -60; la <= 60; la += 30) { const pts = []; for (let lo = -180; lo <= 180; lo += 5) pts.push([la, lo]); seg(pts); }
  ctx.stroke();
  const t0 = performance.now(), ras = globeRaster(R, dpr, sT, cT, sL, cL, sky ? sky.sun : null);
  ctx.imageSmoothingEnabled = true; ctx.drawImage(ras, cx - R, cy - R, 2 * R, 2 * R);
  GL.cost = performance.now() - t0;
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.strokeStyle = C("--ink"); ctx.lineWidth = 1.5; ctx.stroke();
  /* on yarim: halkalar, uydular, etiketler */
  const labs = [];
  ringPaths.forEach(({ rg, main, ghost }) => { ctx.strokeStyle = C(rg.cv);
    if (ghost) { ctx.setLineDash([5, 6]); ctx.globalAlpha = .6; ctx.lineWidth = 1.3; ctx.stroke(ghost[0]); ctx.setLineDash([]); }
    ctx.globalAlpha = 1; ctx.lineWidth = 2; ctx.stroke(main[0]);
    proj(rg.lab[0], rg.lab[1], rg.r); if (P[2] > .1) labs.push([rg.nm + " " + ft(rg.lt), C(rg.cv), P[0], P[1]]); });
  if (sky) sky.sats.forEach(s => sat(s, true));
  ctx.font = "600 14px " + C("--f-ui"); ctx.lineJoin = "round";
  labs.forEach(([txt, col, x, y]) => { ctx.lineWidth = 5; ctx.strokeStyle = "#080b14"; ctx.strokeText(txt, x + 10, y - 6); ctx.fillStyle = col; ctx.fillText(txt, x + 10, y - 6); });
  /* secilen yer */
  if (GL.pick) { proj(GL.pick[0], GL.pick[1]); if (P[2] > 0) { ctx.beginPath(); ctx.arc(P[0], P[1], 9, 0, 2 * Math.PI); ctx.strokeStyle = C("--ground"); ctx.lineWidth = 5; ctx.stroke();
    ctx.strokeStyle = C("--ink"); ctx.lineWidth = 2; ctx.stroke(); ctx.beginPath(); ctx.arc(P[0], P[1], 2.2, 0, 2 * Math.PI); ctx.fillStyle = C("--ink"); ctx.fill(); } }
}
function globeClock() { const fr = GL.t - Math.floor(GL.t), m = Math.floor(fr * 1440), s = "UTC " + String(Math.floor(m / 60)).padStart(2, "0") + ":" + String(m % 60).padStart(2, "0");
  if (s !== GL.clk) { GL.clk = s; $("gclock").textContent = s; } }
function globeTick(t) {
  GL.raf = 0; if (PAGE !== "ana" || !GL.on || document.hidden) return;
  const dt = Math.min(.1, (t - (GL.last || t)) / 1000); GL.last = t;
  const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (GL.auto && !GL.spin && !GL.drag && !GL.pick && !calm && t - GL.idle > 5000) GL.spin = true;
  if (GL.spin && !GL.drag && !calm) {
    if (GL.tracks) { const dd = dt / GL.daySec; GL.t += dd; GL.lon -= 360 * dd; globeClock(); }      /* bir gun = GL.daySec saniye; kamera Gunes'e gore sabit: Dunya altta doner */
    else GL.lon += 5 * dt; }
  if (GL.play && GL.mode === "ay") { GL.mf = (GL.mf + dt / 1.1) % 12; const mo = Math.floor(GL.mf); if (mo !== GL.shown) { GL.shown = mo; $("gmon").value = mo; $("gmlab").textContent = MONTHS[mo]; } }
  if (typeof tourTick === "function") tourTick(dt);
  const moving = (GL.spin && !calm) || !!GL.drag || (GL.play && GL.mode === "ay") || TOUR.on;
  /* hareket ederken kare suresi uzarsa cozunurlugu dusur; durunca tam cozunurlukte bir kez daha ciz */
  if (moving) { if (GL.cost > 20 && GL.q > .5) GL.q = Math.max(.5, GL.q - .25); } else GL.q = 1;
  if (moving || GL.dirty || GL.wasMoving) globeDraw();
  GL.wasMoving = moving; GL.dirty = false;
  if (moving || (GL.auto && !GL.spin && !GL.pick && !calm)) GL.raf = requestAnimationFrame(globeTick);
}
function globeKick() { GL.dirty = true; if (!GL.raf) { GL.last = 0; GL.raf = requestAnimationFrame(globeTick); } }
function globeUI() {
  const H = D.hero || {}, dr = D.drift24 || D.drift;
  $("hv-kure").setAttribute("aria-pressed", GL.on); $("hv-duz").setAttribute("aria-pressed", !GL.on);
  $("globe").hidden = !GL.on; $("herosvg").hidden = GL.on; $("gmodes").hidden = !GL.on; $("grow2").hidden = !GL.on;
  $("globe").closest(".hero").classList.toggle("space", GL.on);
  document.querySelectorAll("#gmodes [data-gm]").forEach(bt => { bt.setAttribute("aria-pressed", bt.dataset.gm === GL.mode);
    bt.hidden = (bt.dataset.gm === "ay" && !H.m) || (bt.dataset.gm === "degisim" && !H.tr) || (bt.dataset.gm === "son" && !H.rc); });
  $("gbar").hidden = !(GL.on && GL.mode === "ay" && H.m);
  $("gtrk").setAttribute("aria-pressed", GL.tracks);
  $("gpp").textContent = GL.auto ? "Durdur" : "Döndür"; $("gclock").hidden = !GL.tracks;
  const hasYears = !!(dr && dr.years && dr.years.length > 1);
  $("gyear").hidden = !(GL.on && GL.tracks && hasYears);
  if (hasYears) { const yr = $("gyr"); yr.max = dr.years.length - 1; if (GL.yi == null) GL.yi = dr.years.length - 1; yr.value = GL.yi; $("gyrlab").textContent = dr.years[GL.yi]; }
  $("gplay").textContent = GL.play ? "Durdur" : "Oynat"; const mo = Math.floor(GL.mf) % 12; $("gmon").value = mo; $("gmlab").textContent = MONTHS[mo];
  $("gpick").hidden = !(GL.on && GL.pick) || TOUR.on; $("tourgo").hidden = !GL.on || TOUR.on; if (!GL.on && TOUR.on) { TOUR.on = false; $("tour").hidden = true; }
  if (GL.pick) { const [la, lo] = GL.pick; $("gpickll").textContent = `${nf(Math.abs(la), 1)}° ${la >= 0 ? "K" : "G"}, ${nf(Math.abs(lo), 1)}° ${lo >= 0 ? "D" : "B"}`; }
  if (!GL.on) return;
  globeClock();
  const key = $("herokey"), grad = (GL.mode === "degisim" && H.tr) ? [gcss("--viirs"), gcss("--ground3"), gcss("--harm")] : ["--t0", "--t1", "--t2", "--t3", "--t4"].map(gcss);
  const txt = GL.mode === "ay" && H.m ? `Ay ay yangın: ${H.mY[0]}–${H.mY[1]} toplamı. Yangın mevsiminin yıl içinde dünyayı dolaşmasını izle.`
    : GL.mode === "degisim" && H.tr ? `Mavi: yangın azaldı, kırmızı: arttı. ${H.trY[2]}–${H.trY[3]} dönemi, ${H.trY[0]}–${H.trY[1]} dönemine göre (aynı sensör: VIIRS S-NPP).`
    : GL.mode === "son" && H.rc ? `${dSpan(H.rc.bas, H.rc.son)}: son ${H.rc.gun} günde uyduların gördüğü yangınlar (VIIRS NOAA-20 ve NOAA-21).`
    : `Küreyi ${H.y0}–${H.y1} arasında VIIRS'in gördüğü yangınlar çiziyor. Sürükleyerek döndür, bir yere tıklayarak seç.`;
  const trkTxt = GL.tracks && hasYears ? ` Halkalar: uyduların ${dr.years[GL.yi]} yörüngeleri, gerçek yükseklikte; kesikli halka ${dr.years[0]} yörüngesi. Gölge: gece.` : "";
  key.innerHTML = `<i style="background:linear-gradient(90deg,${grad.join(",")})"></i><span>${txt}${trkTxt}</span>`;
}
function globeShow() { globeInit(); globeUI(); if (GL.on) { const c = $("globe"), w = c.parentElement.clientWidth; c.style.height = Math.max(380, Math.min(660, Math.round(w * .42))) + "px"; globeDraw(); globeKick(); } }
/* ================= REHBERLI TUR (sunum modu) =================
   Dort adimda projenin hikayesi; her adim kureyi ayarlar ve bir sure sonra kendiliginden ilerler. Sayilar veriden gelir. */
const TOUR = { on: false, i: 0, left: 0, yf: null, fly: null, auto: true };
function tourSteps() {
  const dr = D.drift24 || D.drift, W = D.driftWorld, tr = (D.evidence || []).find(e => e.key === "TR");
  const f = a => { const i = (a || []).findIndex(v => v != null); return i < 0 ? null : a[i]; }, l = a => { const b = (a || []).filter(v => v != null); return b.length ? b[b.length - 1] : null; };
  let p2 = "Uyduların yakıtı bitiyor, yörüngeleri kayıyor.";
  if (dr && f(dr.terra) != null) p2 = `Terra'nın geçişi ${ft(f(dr.terra))} iken ${ft(l(dr.terra))} oldu; Aqua'nınki ${ft(f(dr.aqua))} iken ${ft(l(dr.aqua))}. Kesikli halka ${dr.years[0]} yörüngesi.`;
  if (W && W.terra && W.terra.length > 1) { const i0 = W.terra.findIndex(v => v > 0), ch = (W.terra[W.terra.length - 1] / W.terra[i0] - 1) * 100;
    if (ch < 0) p2 += ` Terra artık yangınlar başlamadan geçiyor: dünya genelinde gördüğü yangın %${nf(Math.abs(ch))} azaldı.`; }
  let p3 = "Terra 2027 başında, Aqua 2027 sonbaharında kapanıyor. Kaydı VIIRS sürdürecek, ama aynı yangını 4–5 kat fazla sayıyor. İki sayımı aynı ölçeğe çeviriyoruz";
  p3 += tr && tr.isabet ? `: modelin görmediği yıllarda Türkiye'nin yıllık toplamını %${tr.isabet} isabetle tutturuyor.` : ".";
  return [
    { h: "Yirmi yıllık yangın kaydı", p: "MODIS 2000'den beri Dünya'daki yangınları sayıyor. Bu kürede kıyı çizgisi yok: haritayı yalnızca uyduların gördüğü yangınlar çiziyor.", sec: 9,
      go: () => { GL.mode = "toplam"; GL.play = false; GL.tracks = true; GL.yi = 0; GL.pick = null; GL.auto = GL.spin = true; GL.lat = 18; GL.lon = 180 - 360 * (GL.t - Math.floor(GL.t)) + 8; } },
    { h: "Saat kayıyor, sayım da kayıyor", p: p2, sec: 12,
      go: () => { GL.mode = "toplam"; GL.play = false; GL.tracks = true; GL.pick = null; GL.auto = GL.spin = true; GL.yi = 0; TOUR.yf = 0; } },
    { h: "Kayıt VIIRS'e geçiyor", p: p3, sec: 12,
      go: () => { TOUR.yf = null; if (dr && dr.years) GL.yi = dr.years.length - 1; GL.pick = null; GL.auto = GL.spin = true;
        if (D.hero && D.hero.m) { GL.mode = "ay"; GL.play = !matchMedia("(prefers-reduced-motion: reduce)").matches; GL.mf = 0; } } },
    { h: "Sıra sende", p: "Kürede bir yere tıkla ya da bir yer ara: o yerin 2003'ten bugüne uyumlu yangın takvimi saniyeler içinde açılır.", sec: 0, last: true,
      go: () => { GL.mode = "toplam"; GL.play = false; GL.auto = GL.spin = false; GL.pick = [39, 33]; TOUR.fly = { lon: 33, lat: 30 }; } }];
}
function tourShow(i) {
  const st = tourSteps(); TOUR.i = Math.max(0, Math.min(st.length - 1, i)); const s = st[TOUR.i];
  TOUR.fly = null; s.go(); TOUR.left = s.sec; TOUR.total = s.sec;
  $("tourn").textContent = `${TOUR.i + 1} / ${st.length}`; $("tourh").textContent = s.h; $("tourp").textContent = s.p;
  $("tourprev").disabled = TOUR.i === 0; $("tournext").textContent = s.last ? "Türkiye'nin takvimini aç" : "İleri";
  $("tourbar").style.width = "0%"; globeUI(); globeKick();
}
function tourStart() { GL.on = true; TOUR.on = true; $("tour").hidden = false; $("tourgo").hidden = true; render(); tourShow(0); }
function tourEnd() { TOUR.on = false; TOUR.yf = null; TOUR.fly = null; $("tour").hidden = true; $("tourgo").hidden = !GL.on; GL.auto = true; GL.idle = performance.now(); globeUI(); globeKick(); }
function tourNext() { const st = tourSteps();
  if (st[TOUR.i].last) { tourEnd(); shareOpenFromTour(); return; }
  tourShow(TOUR.i + 1); }
function shareOpenFromTour() { const q = { kutu: "25.000,35.000,45.000,43.000", ad: "Türkiye ve çevresi", y0: 2003, y1: new Date().getUTCFullYear() };
  wProbe().then(ok => { if (ok) { SHARE_SEEN = shareHash(q).split("?")[1]; location.hash = shareHash(q); shareOpen(q); } else { location.hash = "#/hesapla"; } }); }
function tourTick(dt) {
  if (!TOUR.on) return;
  const dr = D.drift24 || D.drift;
  if (TOUR.yf != null && dr && dr.years) { const n = dr.years.length - 1; TOUR.yf = Math.min(n, TOUR.yf + dt * n / 7); const yi = Math.round(TOUR.yf);
    if (yi !== GL.yi) { GL.yi = yi; $("gyr").value = yi; $("gyrlab").textContent = dr.years[yi]; } if (TOUR.yf >= n) TOUR.yf = null; }
  if (TOUR.fly) { const k = Math.min(1, dt * 2.2), dl = ((TOUR.fly.lon - GL.lon + 540) % 360) - 180; GL.lon += dl * k; GL.lat += (TOUR.fly.lat - GL.lat) * k;
    if (Math.abs(dl) < .2 && Math.abs(TOUR.fly.lat - GL.lat) < .2) TOUR.fly = null; }
  if (TOUR.total > 0 && TOUR.auto) { TOUR.left -= dt; $("tourbar").style.width = Math.max(0, Math.min(100, 100 * (1 - TOUR.left / TOUR.total))).toFixed(1) + "%"; if (TOUR.left <= 0) tourShow(TOUR.i + 1); }
}
function tourInit() {
  $("tourgo").addEventListener("click", tourStart); $("tourend").addEventListener("click", tourEnd);
  $("tournext").addEventListener("click", tourNext); $("tourprev").addEventListener("click", () => tourShow(TOUR.i - 1));
}
function dSpan(a, b) {               /* "2026-11-02","2026-11-13" -> "2–13 Kasım 2026" */
  const [ya, ma, da] = a.split("-").map(Number), [yb, mb, db] = b.split("-").map(Number);
  return ya === yb && ma === mb ? `${da}–${db} ${MONTHS[mb - 1]} ${yb}` : `${da} ${MONTHS[ma - 1]} – ${db} ${MONTHS[mb - 1]} ${yb}`;
}
const llTxt = (la, lo) => `${nf(Math.abs(la), 1)}° ${la >= 0 ? "K" : "G"}, ${nf(Math.abs(lo), 1)}° ${lo >= 0 ? "D" : "B"}`;
let NOW_NAMES = null;
function nowTable() {                 /* ana sayfa: son gunlerde en cok yanan yerler */
  const rc = D.hero && D.hero.rc, sec = $("nowsec"); if (!rc || !rc.top || !rc.top.length) { sec.hidden = true; return; }
  sec.hidden = false;
  $("nowcap").textContent = `${dSpan(rc.bas, rc.son)} arasında VIIRS'in (NOAA-20 ve NOAA-21) en çok yangın gördüğü yerler.` +
    (rc.by ? ` Normal: aynı günlerin ${rc.by[0]}–${rc.by[1]} ortalaması (VIIRS S-NPP).` : "") + " Sayılar 3°×3°'lik alan içindir.";
  const t = $("nowtab");
  t.innerHTML = "<tr><th>Yer</th><th>Tespit</th><th>Normale göre</th><th></th></tr>" + rc.top.map((r, i) => {
    const nm = NOW_NAMES && NOW_NAMES[i];
    return `<tr><td class="sat"><button type="button" class="nowgo" data-i="${i}">${nm ? esc(nm) : llTxt(r.la + .5, r.lo + .5)}</button>${nm ? `<small>${llTxt(r.la + .5, r.lo + .5)}</small>` : ""}</td>` +
      `<td>${nf(r.v)}</td><td class="${r.x != null && r.x >= 1.5 ? "moved" : ""}">${r.x == null ? "normalde az yanar" : nf(r.x, 1) + " kat"}</td>` +
      `<td><button type="button" class="mini nowcalc" data-i="${i}">takvimi aç</button></td></tr>`; }).join("");
  t.querySelectorAll(".nowgo").forEach(b => b.addEventListener("click", () => { const r = rc.top[+b.dataset.i];
    GL.on = true; GL.mode = "son"; GL.spin = false; GL.idle = performance.now(); GL.lon = r.lo + .5; GL.lat = Math.max(-60, Math.min(60, r.la + .5)); GL.pick = [r.la + .5, r.lo + .5];
    render(); $("globe").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" }); }));
  t.querySelectorAll(".nowcalc").forEach(b => b.addEventListener("click", () => { const r = rc.top[+b.dataset.i]; globeJump(r.la + .5, r.lo + .5); }));
  if (NOW_NAMES === null) { NOW_NAMES = [];                       /* yer adlari: sunucu varsa, bir kez */
    wProbe().then(async ok => { if (!ok) return;
      for (let i = 0; i < rc.top.length; i++) { const r = rc.top[i];
        try { const p = await api(`/api/ters?lat=${(r.la + .5).toFixed(2)}&lon=${(r.lo + .5).toFixed(2)}&duzey=il`);
          if (p && p.tur !== "kutu" && p.ad) { NOW_NAMES[i] = p.ad + (p.ulke && p.ulke !== p.ad ? ", " + p.ulke : ""); if (PAGE === "ana") nowTable(); } } catch (e) {} } }); }
}
async function globeJump(lat, lon) {   /* kurede secilen yerin takvimi: Hesapla > Dunyada ara > il duzeyinde sinir > hesapla */
  const msg = $("gpickmsg"); msg.textContent = "";
  if (!(await wProbe())) { msg.textContent = "Bunun için sunucu gerekir: python sunucu.py"; return; }
  const q = { lat: lat.toFixed(3), lon: lon.toFixed(3), duzey: "il", y0: 2015, y1: new Date().getUTCFullYear() };
  SHARE_SEEN = shareHash(q).split("?")[1]; location.hash = shareHash(q); await shareOpen(q);
}
/* ================= PAYLASILABILIR BAGLANTI =================
   Secilen yer ve yillar adres cubuguna yazilir (#/hesapla?...). Baglantiyi acan ayni hesabi gorur.
   Idari sinirlar adresle tasinmaz: yer, merkez noktasi + duzey (ulke/il/ilce) ya da kutu olarak yazilir ve acilista yeniden bulunur. */
const SHARE_LV = { country: "ulke", state: "il", province: "il", region: "il", county: "ilce", district: "ilce", municipality: "ilce", city: "ilce", town: "ilce" };
function shareOf(p, y0, y1) {
  const q = { y0: Math.min(y0, y1), y1: Math.max(y0, y1) }, lv = SHARE_LV[p.tur];
  if (lv && p.lat != null && p.lon != null) { q.lat = (+p.lat).toFixed(3); q.lon = (+p.lon).toFixed(3); q.duzey = lv; }
  else { const [w, s, e, n] = p.bbox || geoBBox(p.geojson); q.kutu = [w, s, e, n].map(v => (+v).toFixed(3)).join(","); }
  if (p.ad) q.ad = String(p.ad).slice(0, 60);
  return q;
}
function shareHash(q) { return "#/hesapla?" + new URLSearchParams(q).toString(); }
function shareSet(q) { if (!q || SHARE_BUSY) return; SHARE_SEEN = shareHash(q).split("?")[1]; if (PAGE === "hesapla") try { history.replaceState(null, "", shareHash(q)); } catch (e) {} }
let SHARE_SEEN = null, SHARE_BUSY = false;
async function shareOpen(q) {          /* baglantidaki (ya da kurede secilen) yeri ac ve hesapla */
  SHARE_BUSY = true;
  try { await shareOpen2(q); } finally { SHARE_BUSY = false; if (REG && REG.share) shareSet(REG.share); }
}
async function shareOpen2(q) {
  if (q.bolge) { const r = D.regions.find(x => x.key === q.bolge); if (r) { $("mode-region").click(); $("exreg").value = r.key; await useRegion(r.key); } return; }
  if (!(await wProbe())) { const m = $("excal"); if (m) { m.textContent = "Bu bağlantıyı açmak için sunucu gerekir: python sunucu.py"; m.className = "msg err"; } return; }
  $("mode-world").click();
  try {
    await wMap(); if (!WSETS.length) await wSets();
    const yNow = new Date().getUTCFullYear(), y0 = Math.max(2003, Math.min(yNow, +q.y0 || 2016)), y1 = Math.max(2003, Math.min(yNow, +q.y1 || yNow));
    $("wy0").value = Math.min(y0, y1); $("wy1").value = Math.max(y0, y1);
    let p;
    if (q.kutu) { const [w, s, e, n] = q.kutu.split(",").map(Number); if ([w, s, e, n].some(v => !isFinite(v)) || e <= w || n <= s) throw new Error("Bağlantıdaki kutu okunamadı.");
      p = { ad: q.ad || "Çizilen kutu", tam: `${nf(s, 2)}…${nf(n, 2)}° enlem · ${nf(w, 2)}…${nf(e, 2)}° boylam`, tur: "kutu", bbox: [w, s, e, n], lat: (s + n) / 2, lon: (w + e) / 2,
        geojson: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } }; }
    else { const la = +q.lat, lo = +q.lon; if (!isFinite(la) || !isFinite(lo) || Math.abs(la) > 90 || Math.abs(lo) > 180) throw new Error("Bağlantıdaki konum okunamadı.");
      wMsg("sınır aranıyor…"); p = await api(`/api/ters?lat=${la.toFixed(4)}&lon=${lo.toFixed(4)}&duzey=${({ ulke: "ulke", il: "il", ilce: "ilce" })[q.duzey] || "il"}`); }
    wSelect(p); wMsg(""); await wRun();
  } catch (e) { wMsg(e.message, true); }
}
function shareRoute() {                /* adres cubugundaki sorgu degistiyse ac */
  const qs = (location.hash.split("?")[1] || ""); if (!qs || qs === SHARE_SEEN) return; SHARE_SEEN = qs;
  const q = Object.fromEntries(new URLSearchParams(qs)); if (q.bolge || q.kutu || (q.lat && q.lon)) shareOpen(q);
}
async function shareCopy() {
  const bt = $("exshare"), q = REG && REG.share; if (!q) return;
  const url = location.origin + location.pathname + location.search + shareHash(q);
  let ok = false; try { await navigator.clipboard.writeText(url); ok = true; } catch (e) {
    const ta = document.createElement("textarea"); ta.value = url; ta.style.position = "fixed"; ta.style.opacity = "0"; document.body.appendChild(ta); ta.select();
    try { ok = document.execCommand("copy"); } catch (e2) {} ta.remove(); }
  bt.textContent = ok ? "Kopyalandı" : "Kopyalanamadı"; setTimeout(() => { bt.textContent = "Bağlantıyı kopyala"; }, 1600);
}
/* ================= RAPOR (PDF) =================
   Sonuc sayfasi yazdirma duzenine gecer (acik tema, A4 genisligi), tarayicinin "PDF olarak kaydet" secenegiyle tek dosya olur. */
function exReport() {
  if (!RES || !REG) return;
  const root = document.documentElement, prev = root.getAttribute("data-theme"), today = new Date();
  $("reph").textContent = REG.name || "Seçilen alan";
  $("repp").textContent = `Ateş Takvimi yangın raporu. Dönem ${Y0 === Y1 ? Y0 : Y0 + "–" + Y1}. Hazırlanma tarihi ${today.getDate()} ${MONTHS[today.getMonth()]} ${today.getFullYear()}.`;
  document.body.classList.add("printing"); root.setAttribute("data-theme", "light"); render();
  const done = () => { removeEventListener("afterprint", done); document.body.classList.remove("printing"); if (prev) root.setAttribute("data-theme", prev); else root.removeAttribute("data-theme"); render(); };
  addEventListener("afterprint", done);
  requestAnimationFrame(() => requestAnimationFrame(() => { try { window.print(); } catch (e) { done(); } }));
}
/* ================= SINIR DOSYASI (GeoJSON / KML) =================
   Kullanici kendi sinirini yukler (orman isletmesi, milli park, havza); hesap o sinirin icindeki hucrelerle yapilir. */
function parseBoundary(text, name) {
  const polys = [];                    /* her biri: [dis halka, ic halkalar...] ; halka: [[boylam, enlem], ...] */
  const ring = arr => arr.map(p => [+p[0], +p[1]]).filter(p => isFinite(p[0]) && isFinite(p[1]));
  const t = text.trim();
  if (t[0] === "{" || t[0] === "[") {
    const walk = g => { if (!g) return;
      if (g.type === "FeatureCollection") (g.features || []).forEach(walk);
      else if (g.type === "Feature") walk(g.geometry);
      else if (g.type === "GeometryCollection") (g.geometries || []).forEach(walk);
      else if (g.type === "Polygon") polys.push(g.coordinates.map(ring));
      else if (g.type === "MultiPolygon") g.coordinates.forEach(pl => polys.push(pl.map(ring))); };
    walk(JSON.parse(t));
  } else {
    const doc = new DOMParser().parseFromString(t, "application/xml");
    if (doc.querySelector("parsererror")) throw new Error("Dosya okunamadı: geçerli bir KML ya da GeoJSON değil.");
    const coords = el => ring((el.textContent || "").trim().split(/\s+/).map(s => s.split(",")));
    [...doc.getElementsByTagNameNS("*", "Polygon")].forEach(pg => {
      const rings = []; const outer = pg.getElementsByTagNameNS("*", "outerBoundaryIs")[0];
      if (outer) { const c = outer.getElementsByTagNameNS("*", "coordinates")[0]; if (c) rings.push(coords(c)); }
      [...pg.getElementsByTagNameNS("*", "innerBoundaryIs")].forEach(ib => { const c = ib.getElementsByTagNameNS("*", "coordinates")[0]; if (c) rings.push(coords(c)); });
      if (rings.length) polys.push(rings); });
  }
  let good = polys.map(pl => pl.filter(r => r.length >= 4)).filter(pl => pl.length);
  if (!good.length) throw new Error("Dosyada alan (Polygon) bulunamadı. Nokta ve çizgiler kullanılamaz.");
  let total = good.reduce((a, pl) => a + pl.reduce((b, r) => b + r.length, 0), 0);
  if (total > 40000) { const k = Math.ceil(total / 40000);         /* cok ayrintili sinir: her k noktadan birini al */
    good = good.map(pl => pl.map(r => { const o = r.filter((p, i) => i % k === 0); if (o.length && (o[0][0] !== o[o.length - 1][0] || o[0][1] !== o[o.length - 1][1])) o.push(o[0]); return o; }).filter(r => r.length >= 4)).filter(pl => pl.length); }
  let w = 180, s = 90, e = -180, n = -90;
  good.forEach(pl => pl[0].forEach(([x, y]) => { if (x < w) w = x; if (x > e) e = x; if (y < s) s = y; if (y > n) n = y; }));
  if (w < -180 || e > 180 || s < -90 || n > 90) throw new Error("Koordinatlar enlem-boylam (WGS84) değil. Dosyayı EPSG:4326 olarak dışa aktar.");
  const geojson = good.length === 1 ? { type: "Polygon", coordinates: good[0] } : { type: "MultiPolygon", coordinates: good };
  const ad = name.replace(/\.[^.]+$/, "").slice(0, 60) || "Yüklenen sınır";
  return { ad, tam: `${name}: ${good.length} alan, ${nf(Math.min(total, 40000))} nokta`, tur: "dosya", bbox: [w, s, e, n], lat: (s + n) / 2, lon: (w + e) / 2, geojson };
}
function boundaryInit() {
  const inp = $("wbfile"); if (!inp) return;
  inp.addEventListener("change", async () => { const f = inp.files[0]; if (!f) return;
    try { if (f.size > 20e6) throw new Error("Dosya çok büyük (en fazla 20 MB).");
      const p = parseBoundary(await f.text(), f.name); wSelect(p); wMsg(`Sınır yüklendi: ${p.tam}. Yılları seç, sonra Takvimi hesapla'ya bas.`); }
    catch (e) { wMsg(e.message, true); }
    inp.value = ""; });
}
function exInit() {
  const sel = document.getElementById("exreg");
  D.regions.forEach(r => sel.add(new Option(r.name, r.key)));
  sel.addEventListener("change", () => useRegion(sel.value));
  ["exy0", "exy1", "exlayer"].forEach(id => document.getElementById(id).addEventListener("change", exRender));
  document.getElementById("exclear").addEventListener("click", () => { SEL = null; exRender(); });
  document.getElementById("exdl").addEventListener("click", exDownload);
  document.getElementById("exrep").addEventListener("click", exReport);
  document.getElementById("exshare").addEventListener("click", shareCopy);
  boundaryInit();
  const setMode = async m => { MODE = m; ["region", "world", "upload"].forEach(k => document.getElementById("mode-" + k).setAttribute("aria-pressed", m === k));
    document.getElementById("panel-upload").hidden = m !== "upload"; document.getElementById("panel-world").hidden = m !== "world";
    const wasHidden = document.querySelector(".work").hidden;
    showOut(!(m === "world" && SERVER && !(REG && REG.boundary)));
    if (wasHidden && !document.querySelector(".work").hidden) exRender();
    if (m === "world" && await wProbe()) { showOut(!!(REG && REG.boundary)); wMap(); if (!WSETS.length) wSets(); } };
  ["region", "world", "upload"].forEach(k => document.getElementById("mode-" + k).addEventListener("click", () => setMode(k)));
  wInit();
  wProbe().then(ok => { if (ok && !/[?&]bolge=/.test(location.hash)) setMode("world"); });
  document.getElementById("exfile").addEventListener("change", e => e.target.files.length && onFiles([...e.target.files]));
  const drop = document.getElementById("panel-upload");
  drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", e => { e.preventDefault(); drop.classList.remove("over"); if (e.dataTransfer.files.length) onFiles([...e.dataTransfer.files]); });
  if (D.regions.length && !/[?&]bolge=/.test(location.hash)) useRegion(D.regions[0].key);
}

/* ---------- ipucu ---------- */
const TIP = document.getElementById("tip");
document.addEventListener("pointermove", e => {
  const t = e.target.closest && e.target.closest("[data-tip]");
  if (!t) { TIP.hidden = true; return; }
  TIP.innerHTML = t.getAttribute("data-tip"); TIP.hidden = false;
  const x = Math.min(e.clientX, innerWidth - 280), y = Math.min(e.clientY, innerHeight - 90);
  TIP.style.left = x + "px"; TIP.style.top = y + "px";
});
document.addEventListener("pointerleave", () => TIP.hidden = true);

/* ================= KARSILASTIR ================= */
let CMODE = "yer";
const CMPC = () => [css("--harm"), css("--viirs")];
async function serverRegion(place, y0, y1, onMsg) {
  /* sunucudan bir yerin uyumlu verisini al (hazir veride aninda) */
  const { id } = await api("/api/bolge", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ yer: place, y0, y1 }) });
  let st;
  for (;;) { await new Promise(r => setTimeout(r, 350)); st = await api(`/api/is?id=${id}`); if (onMsg) onMsg(st.mesaj); if (st.durum === "bitti" || st.durum === "hata") break; }
  if (st.durum === "hata") throw new Error(st.mesaj);
  return api(`/api/sonuc?id=${id}`);
}
const boxPlace = d => { const [w, s, e, n] = d.kutu; return { ad: d.ad, tam: d.olay || "", tur: "indirilen bölge", bbox: d.kutu, lat: (s + n) / 2, lon: (w + e) / 2,
  geojson: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } }; };
function coverSet(place, y0, y1) {
  /* bu yeri kapsayan en kucuk hazir veri seti (yil araligi verilirse onu da kapsamali) */
  const [w, s, e, n] = place.bbox || geoBBox(place.geojson);
  return WSETS.filter(d => d.kutu[0] <= w + 1e-6 && d.kutu[1] <= s + 1e-6 && d.kutu[2] >= e - 1e-6 && d.kutu[3] >= n - 1e-6 && (y0 == null || (d.y0 <= y0 && d.y1 >= y1)))
    .sort((p, q) => (p.kutu[2] - p.kutu[0]) * (p.kutu[3] - p.kutu[1]) - (q.kutu[2] - q.kutu[0]) * (q.kutu[3] - q.kutu[1]))[0];
}
async function getReg(key) {
  if (key === "__upload") return UPLOADED;
  if (REGS[key]) return REGS[key];
  if (key.startsWith("DS:")) {
    const d = WSETS.find(x => x.key === key.slice(3)); if (!d) return null;
    $("cmpnote").textContent = `${d.ad} hazırlanıyor…`;
    try { const obj = await serverRegion(boxPlace(d), d.y0, d.y1); obj.key = key; REGS[key] = obj; $("cmpnote").textContent = ""; }
    catch (e) { $("cmpnote").textContent = e.message; return null; }
    return REGS[key];
  }
  const r = D.regions.find(x => x.key === key); if (!r) return null; REGS[key] = await unpack(r.b64);
  return REGS[key];
}
function cmpOptions() {
  const opts = D.regions.map(r => [r.key, r.name]), have = new Set(D.regions.map(r => r.key));
  WSETS.forEach(d => { if (d.key !== "DUNYA" && !have.has(d.key)) opts.push(["DS:" + d.key, d.ad]); });
  Object.values(REGS).forEach(r => { if (r && r.key && r.key[0] === "Q") opts.push([r.key, `${r.name}${r.donem ? ", " + r.donem : ""}`]); });
  if (UPLOADED) opts.push(["__upload", "Yüklenen dosyalar"]);
  ["ca-r", "cb-r"].forEach((id, k) => { const sel = $(id), cur = sel.value; sel.innerHTML = "";
    opts.forEach(([v, t]) => sel.add(new Option(t, v)));
    sel.value = opts.some(o => o[0] === cur) ? cur : (opts[k] || opts[0] || [""])[0]; });
}
function analyze(R, y0, y1) {
  const save = [REG, Y0, Y1, SEL, RES];
  REG = R; Y0 = y0; Y1 = y1; SEL = null; RES = compute();
  const st = seasonStats(), res = RES;
  [REG, Y0, Y1, SEL, RES] = save;
  const mon = new Float64Array(12), doy = new Float64Array(366);
  for (let j = 0; j < res.nd; j++) { const d = res.d0 + j, t = dDate(d); mon[t.getUTCMonth()] += res.E[j]; doy[Math.min(365, d - yStart(t.getUTCFullYear()))] += res.E[j]; }
  const tot = mon.reduce((a, v) => a + v, 0), ny = y1 - y0 + 1;
  const cum = new Float64Array(366); let c = 0; for (let k = 0; k < 366; k++) { c += doy[k]; cum[k] = tot > 0 ? 100 * c / tot : 0; }
  const at = q => { const k = cum.findIndex(v => v >= q); return k < 0 ? null : k; };
  let peak = null, best = -1; for (let k = 0; k < 366; k++) { let wsum = 0; for (let q = Math.max(0, k - 7); q <= Math.min(365, k + 7); q++) wsum += doy[q]; if (wsum > best) { best = wsum; peak = k; } }
  const ys = st.map(r => r.tot), mean = ys.reduce((a, v) => a + v, 0) / Math.max(ys.length, 1);
  const cv = ys.length > 1 ? Math.sqrt(ys.reduce((a, v) => a + (v - mean) ** 2, 0) / (ys.length - 1)) / Math.max(mean, 1e-9) : null;
  const topM = [...mon.keys()].sort((a, b2) => mon[b2] - mon[a])[0];
  return { R, y0, y1, ny, st, tot, mean: tot / ny, mon: Array.from(mon, v => tot > 0 ? 100 * v / tot : 0), cum, s10: at(10), s50: at(50), s90: at(90), peak, cv, topM: tot > 0 ? topM : null };
}
const doyTxt = k => { if (k == null) return "–"; const t = new Date(Date.UTC(2001, 0, 1) + k * MS); return t.getUTCDate() + " " + AY[t.getUTCMonth()]; };
function lbl(a) { return CMODE === "donem" ? `${a.y0}–${a.y1}` : `${a.R.name}${a.R.donem ? "" : ""}`; }
function cmpFillYears(prefix, R, keep) {
  const [a, b2] = regYears(R), s0 = $(prefix + "-y0"), s1 = $(prefix + "-y1"), o0 = +s0.value, o1 = +s1.value;
  s0.innerHTML = s1.innerHTML = ""; for (let y = a; y <= b2; y++) { s0.add(new Option(y, y)); s1.add(new Option(y, y)); }
  s0.value = keep && o0 >= a && o0 <= b2 ? o0 : a; s1.value = keep && o1 >= a && o1 <= b2 ? o1 : b2;
}
let CMP_BUSY = false, CMP_INIT = false;
async function cmpSetup(changed) {
  if (!$("ca-r").options.length) cmpOptions();
  const RA = await getReg($("ca-r").value); if (!RA) return;
  if (CMODE === "donem") { $("cb-r").value = $("ca-r").value; $("cb-r").disabled = true; } else $("cb-r").disabled = false;
  $("cb-pick").hidden = CMODE === "donem";
  const RB = await getReg($("cb-r").value); if (!RB) return;
  if (!CMP_INIT || changed === "ca-r" || changed === "mode") cmpFillYears("ca", RA, changed !== "mode" && CMP_INIT);
  if (!CMP_INIT || changed === "cb-r" || changed === "ca-r" || changed === "mode") cmpFillYears("cb", RB, false);
  if (CMODE === "donem" && (changed === "mode" || changed === "ca-r" || !CMP_INIT)) {
    const [a, b2] = regYears(RA), mid = Math.floor((a + b2) / 2);
    $("ca-y0").value = a; $("ca-y1").value = mid; $("cb-y0").value = Math.min(mid + 1, b2); $("cb-y1").value = b2;
  }
  CMP_INIT = true;
  cmpRender();
}
let CMP_SETS = false;
async function cmpRender() {
  if (!CMP_SETS) { CMP_SETS = true; if (await wProbe()) { if (!WSETS.length) await wSets(); cmpOptions(); } 
    $("cmpnote").textContent = SERVER ? "" : "Sunucusuz açılan sayfada yalnızca örnek bölgeler var. Tüm bölgeler ve haritadan seçim için python sunucu.py ile aç."; }
  if (!CMP_INIT) return cmpSetup();
  const RA = await getReg($("ca-r").value), RB = await getReg($("cb-r").value);
  if (!RA || !RB) return;
  const yy = p => [Math.min(+$(p + "-y0").value, +$(p + "-y1").value), Math.max(+$(p + "-y0").value, +$(p + "-y1").value)];
  const A = analyze(RA, ...yy("ca")), B = analyze(RB, ...yy("cb")), [cA, cB] = CMPC();
  const nameA = CMODE === "donem" ? `${A.y0}–${A.y1}` : RA.name, nameB = CMODE === "donem" ? `${B.y0}–${B.y1}` : RB.name;
  // ozet cumle
  let s = "";
  if (A.tot > 0 && B.tot > 0) {
    const r = B.mean / A.mean;
    const rt = Math.abs(r - 1) < .05 ? "neredeyse aynı" : r > 1 ? `%${nf(100 * (r - 1), 0)} daha yüksek` : `%${nf(100 * (1 - r), 0)} daha düşük`;
    s += CMODE === "donem" ? `${RA.name}, ${nameB} dönemi: yıllık ortalama aktivite ${nameA} dönemine göre ${rt}. `
                           : `Yıllık ortalama aktivite B tarafında (${nameB}) A tarafına (${nameA}) göre ${r > 3 || r < 1 / 3 ? nf(r, 2) + " kat" : rt}. `;
    if (A.s10 != null && B.s10 != null) { const d = B.s10 - A.s10; s += Math.abs(d) < 4 ? "Sezon iki tarafta da aşağı yukarı aynı günlerde başlıyor" : `Sezon ${nameB} tarafında ${Math.abs(d)} gün daha ${d < 0 ? "erken" : "geç"} başlıyor`; }
    if (A.s90 != null && B.s90 != null && A.s10 != null && B.s10 != null) { const la = A.s90 - A.s10, lb = B.s90 - B.s10, dl = lb - la; s += Math.abs(dl) < 4 ? " ve benzer uzunlukta sürüyor. " : ` ve ${Math.abs(dl)} gün daha ${dl > 0 ? "uzun" : "kısa"} sürüyor. `; }
    if (A.topM != null && B.topM != null) s += A.topM === B.topM ? `En yoğun ay ikisinde de ${MONTHS[A.topM]}.` : `En yoğun ay ${nameA} için ${MONTHS[A.topM]}, ${nameB} için ${MONTHS[B.topM]}.`;
  } else s = "Seçilen dönemlerden birinde hiç tespit yok. Yıl aralığını değiştir.";
  $("cmpsum").textContent = s;

  // aylarin payi
  { const w = W("cmpmon", 300), h = 260, L = 44, top = 16, bot = 30, sv = canvas("cmpmon", w, h);
    const ym = nice(Math.max(...A.mon, ...B.mon, 5)), cw = (w - L) / 12, y = v => top + (h - top - bot) * (1 - v / ym);
    [0, ym / 2, ym].forEach(v => { E("line", { x1: L, x2: w, y1: y(v), y2: y(v), class: "grid" }, sv); T(sv, L - 6, y(v) + 4, "%" + nf(v), { "text-anchor": "end" }); });
    for (let m = 0; m < 12; m++) {
      [[A, cA, 0], [B, cB, 1]].forEach(([X, col, k]) => { const bw = cw * .34, x0 = L + m * cw + cw * .14 + k * bw;
        const r = E("rect", { x: x0, y: y(X.mon[m]), width: bw - 1, height: h - bot - y(X.mon[m]), fill: col }, sv);
        tip(r, `${k ? nameB : nameA} · ${MONTHS[m]}<b>%${nf(X.mon[m], 1)}</b>yıllık aktivitenin payı`); });
      T(sv, L + m * cw + cw / 2, h - 10, AY[m], { "text-anchor": "middle" });
    } }
  // sezon ilerleyisi
  { const w = W("cmpcum", 300), h = 260, L = 44, R = 16, top = 16, bot = 30, sv = canvas("cmpcum", w, h);
    const x = k => L + (w - L - R) * k / 365, y = v => top + (h - top - bot) * (1 - v / 100);
    [0, 50, 100].forEach(v => { E("line", { x1: L, x2: w - R, y1: y(v), y2: y(v), class: "grid", "stroke-dasharray": v === 50 ? "3 4" : "" }, sv); T(sv, L - 6, y(v) + 4, "%" + v, { "text-anchor": "end" }); });
    [0, 59, 120, 181, 243, 304].forEach((d, k) => T(sv, x(d) + 2, h - 10, AY[k * 2]));
    [[A, cA, nameA], [B, cB, nameB]].forEach(([X, col, nm]) => {
      E("polyline", { points: Array.from(X.cum, (v, k) => `${x(k).toFixed(1)},${y(v).toFixed(1)}`).join(" "), fill: "none", stroke: col, "stroke-width": 2.5 }, sv);
      if (X.s50 != null) { const c = E("circle", { cx: x(X.s50), cy: y(50), r: 5, fill: col, stroke: css("--ground"), "stroke-width": 1.5 }, sv); tip(c, `${nm}<b>${doyTxt(X.s50)}</b>yılın aktivitesinin yarısı bu güne kadar tamamlanıyor`); }
    }); }
  // yillara gore
  { const all = [...new Set([...A.st.map(r => r.y), ...B.st.map(r => r.y)])].sort((a, b2) => a - b2);
    const n = all.length, w = W("cmpyear", 300), rot = n > 1 && (w - 62) / (n - 1) < 36, h = rot ? 280 : 250, L = 54, R = 14, top = 16, bot = rot ? 52 : 30, sv = canvas("cmpyear", w, h);
    const ym = nice(Math.max(...A.st.map(r => r.tot), ...B.st.map(r => r.tot), 1)), x = i => n === 1 ? (L + w - R) / 2 : L + i * (w - L - R) / (n - 1), y = v => top + (h - top - bot) * (1 - v / ym);
    [0, ym / 2, ym].forEach(v => { E("line", { x1: L, x2: w - R, y1: y(v), y2: y(v), class: "grid" }, sv); T(sv, L - 6, y(v) + 4, nf(v), { "text-anchor": "end" }); });
    yearAxis(sv, all.map(yv => ({ y: yv })), x, h, bot, rot);
    [[A, cA, nameA], [B, cB, nameB]].forEach(([X, col, nm]) => {
      const pts = X.st.map(r => `${x(all.indexOf(r.y))},${y(r.tot)}`).join(" ");
      E("polyline", { points: pts, fill: "none", stroke: col, "stroke-width": 2.5 }, sv);
      X.st.forEach(r => { const c = E("circle", { cx: x(all.indexOf(r.y)), cy: y(r.tot), r: 3.5, fill: col }, sv); tip(c, `${nm} · ${r.y}<b>${nf(r.tot)}</b>MODIS-eşdeğeri tespit`); });
    }); }
  // tablo
  const rowsT = [
    ["Dönem", `${A.y0}–${A.y1}`, `${B.y0}–${B.y1}`, ""],
    ["Yıllık ortalama (MODIS-eşdeğeri)", nf(A.mean), nf(B.mean), A.mean > 0 ? nf(B.mean / A.mean, 2) + " kat" : "–"],
    ["En yoğun ay", A.topM != null ? MONTHS[A.topM] : "–", B.topM != null ? MONTHS[B.topM] : "–", ""],
    ["Sezon başı (%10)", doyTxt(A.s10), doyTxt(B.s10), A.s10 != null && B.s10 != null ? (B.s10 - A.s10 > 0 ? "+" : "") + (B.s10 - A.s10) + " gün" : ""],
    ["Sezon ortası (%50)", doyTxt(A.s50), doyTxt(B.s50), A.s50 != null && B.s50 != null ? (B.s50 - A.s50 > 0 ? "+" : "") + (B.s50 - A.s50) + " gün" : ""],
    ["Sezon sonu (%90)", doyTxt(A.s90), doyTxt(B.s90), A.s90 != null && B.s90 != null ? (B.s90 - A.s90 > 0 ? "+" : "") + (B.s90 - A.s90) + " gün" : ""],
    ["En yoğun 15 gün", doyTxt(A.peak), doyTxt(B.peak), ""],
    ["Yıldan yıla değişkenlik", A.cv != null ? "%" + nf(100 * A.cv) : "–", B.cv != null ? "%" + nf(100 * B.cv) : "–", ""]];
  $("cmptab").innerHTML = `<tr><th></th><th class="r" style="color:${cA}">${esc(nameA)}</th><th class="r" style="color:${cB}">${esc(nameB)}</th><th class="r">B − A</th></tr>` +
    rowsT.map(r => `<tr><td>${r[0]}</td><td class="r">${r[1]}</td><td class="r">${r[2]}</td><td class="r">${r[3]}</td></tr>`).join("");
}
/* --- karsilastirma icin yer secici (arama, indirilen bolgeler, harita) --- */
const PK = { slot: null, map: null, layer: null, sel: null, level: "il", busy: false };
function pkMsg(t, err) { const m = $("pk-msg"); m.textContent = t || ""; m.className = "msg" + (err ? " err" : ""); }
function pkSelect(p) {
  PK.sel = p;
  if (PK.map) { if (PK.layer) PK.layer.remove();
    PK.layer = L.geoJSON(p.geojson, { style: { color: css("--harm"), weight: 2, fillColor: css("--harm"), fillOpacity: .07 } }).addTo(PK.map);
    try { PK.map.fitBounds(PK.layer.getBounds(), { padding: [24, 24], maxZoom: 9 }); } catch (e) {} }
  const [w, s, e, n] = p.bbox || geoBBox(p.geojson), ds = coverSet(p);
  $("pk-sel").innerHTML = `<b>${esc(p.ad)}</b>${esc(p.tam || "")} (${TUR[p.tur] || esc(p.tur || "")}, alan ${nf(e - w, 1)}° × ${nf(n - s, 1)}°)`;
  pkMsg(ds ? `Hazır veri: ${ds.ad}, ${ds.y0}–${ds.y1}. Anında gelir.` : "Bu yer indirilen verilerin dışında. Önce Hesapla sayfasında canlı hesapla; sonra burada listede görünür.", !ds);
  $("pk-go").disabled = !ds;
}
async function pkOpen(slot) {
  PK.slot = slot; $("cmp-picker").hidden = false; $("pk-title").textContent = `${slot === "ca" ? "A" : "B"} için yer seç`;
  const ok = await wProbe(); $("pk-off").hidden = ok; $("pk-on").hidden = !ok;
  $("cmp-picker").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest" });
  if (!ok) return;
  if (!WSETS.length) await wSets();
  const box = $("pk-chips"); box.innerHTML = "";
  WSETS.filter(d => d.key !== "DUNYA").forEach(d => { const bt = document.createElement("button"); bt.type = "button"; bt.textContent = d.ad; bt.title = d.olay || "";
    bt.addEventListener("click", () => pkSelect(boxPlace(d))); box.appendChild(bt); });
  try { await loadLeaflet(); } catch (e) { pkMsg(e.message, true); return; }
  if (!PK.map) {
    PK.map = L.map("pk-map", { worldCopyJump: true, minZoom: 2 }).setView([39, 30], 4);
    L.tileLayer(tileUrl(), { maxZoom: 12, attribution: "&copy; OpenStreetMap katkıcıları" }).addTo(PK.map);
    if (wDark()) $("pk-map").classList.add("dark");
    PK.map.on("click", async e => { if (PK.busy) return; pkMsg("sınır aranıyor…");
      try { pkSelect(await api(`/api/ters?lat=${e.latlng.lat}&lon=${e.latlng.lng}&duzey=${PK.level}`)); } catch (er) { pkMsg(er.message, true); } });
  } else PK.map.invalidateSize();
}
async function pkUse() {
  if (!PK.sel || PK.busy) return;
  const ds = coverSet(PK.sel); if (!ds) return;
  PK.busy = true; $("pk-go").disabled = true; pkMsg("hesaplanıyor…");
  try {
    const obj = await serverRegion(PK.sel, ds.y0, ds.y1, pkMsg);
    REGS[obj.key] = obj; cmpOptions(); $(PK.slot + "-r").value = obj.key;
    if (CMODE === "donem") $("cb-r").value = obj.key;
    $("cmp-picker").hidden = true; pkMsg("");
    await cmpSetup(PK.slot + "-r");
  } catch (e) { pkMsg(e.message, true); }
  finally { PK.busy = false; $("pk-go").disabled = !PK.sel; }
}
function pkInit() {
  ["ca", "cb"].forEach(sl => $(sl + "-pick").addEventListener("click", () => pkOpen(sl)));
  $("pk-close").addEventListener("click", () => { $("cmp-picker").hidden = true; });
  $("pk-go").addEventListener("click", pkUse);
  $("pk-form").addEventListener("submit", async e => { e.preventDefault(); const q = $("pk-q").value.trim(); if (!q) return;
    const ol = $("pk-res"); ol.innerHTML = ""; pkMsg("aranıyor…");
    try { const res = await api(`/api/ara?q=${encodeURIComponent(q)}`);
      if (!res.length) { pkMsg(`"${q}" için sonuç yok. Başka yazımla dene.`, true); return; }
      pkMsg("");
      res.forEach((p2, i) => { const li = document.createElement("li"), bt = document.createElement("button"); bt.type = "button";
        bt.innerHTML = `<b>${esc(p2.ad)}</b><em>${TUR[p2.tur] || esc(p2.tur)}</em><small>${esc(p2.tam)}</small>`;
        bt.addEventListener("click", () => { [...ol.querySelectorAll("button")].forEach(x => x.removeAttribute("aria-current")); bt.setAttribute("aria-current", "true"); pkSelect(p2); });
        li.appendChild(bt); ol.appendChild(li); if (i === 0) { bt.setAttribute("aria-current", "true"); pkSelect(p2); } });
    } catch (er) { pkMsg(er.message, true); } });
  document.querySelectorAll("#pk-seg [data-plv]").forEach(bt => bt.addEventListener("click", () => {
    PK.level = bt.dataset.plv; document.querySelectorAll("#pk-seg [data-plv]").forEach(x => x.setAttribute("aria-pressed", x === bt)); }));
}
function cmpInit() {
  pkInit();
  ["yer", "donem"].forEach(m => $("cmp-m-" + m).addEventListener("click", () => { CMODE = m; ["yer", "donem"].forEach(k => $("cmp-m-" + k).setAttribute("aria-pressed", k === m)); cmpSetup("mode"); }));
  ["ca-r", "cb-r"].forEach(id => $(id).addEventListener("change", () => cmpSetup(id)));
  ["ca-y0", "ca-y1", "cb-y0", "cb-y1"].forEach(id => $(id).addEventListener("change", cmpRender));
  cmpOptions();
}

/* ================= SAYFALAR ================= */
const PAGES = { ana: "Ateş Takvimi", hesapla: "Hesapla · Ateş Takvimi", karsilastir: "Karşılaştır · Ateş Takvimi", yontem: "Yöntem ve kanıt · Ateş Takvimi", hakkinda: "Hakkında · Ateş Takvimi" };
let PAGE = "ana";
function route() {
  const p = (location.hash.replace(/^#\/?/, "").split(/[/?]/)[0]) || "ana";
  const prev = PAGE; PAGE = p in PAGES ? p : "ana";
  document.querySelectorAll(".pg").forEach(sx => sx.hidden = sx.dataset.page !== PAGE);
  document.querySelectorAll(".nav a").forEach(a => { if (a.dataset.p === PAGE) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
  document.title = PAGES[PAGE];
  if (prev !== PAGE) scrollTo(0, 0);
  render();
  if (PAGE === "hesapla") shareRoute();
  if (PAGE === "hesapla" && WMAP) setTimeout(() => WMAP.invalidateSize(), 50);
}
function render() {
  STOPS = ["--t0", "--t1", "--t2", "--t3", "--t4"].map(v => hex(css(v)));
  homeText();
  if (PAGE === "ana") { hiLoad(); if (!GL.on || !D.hero) hero(); globeShow(); passTable(); nowTable(); }
  else if (PAGE === "hesapla") exRender();
  else if (PAGE === "karsilastir") cmpRender();
  else if (PAGE === "yontem") { evidence(); line(); steps(); pers(); shiftCard(); dial(); }
}
exInit();
cmpInit();
addEventListener("hashchange", route);
route();
let RT; addEventListener("resize", () => { clearTimeout(RT); RT = setTimeout(render, 150); });
new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
if (document.fonts && document.fonts.ready) document.fonts.ready.then(render);
/* ================= ENGLISH (EN) ================= */
/* Sayfa Turkce uretilir; EN secilince gorunen metinler bu tablolarla cevrilir (sozluk + kaliplar + ifadeler).
   Yeni eklenen dugumler MutationObserver ile cevrilir. TR'ye donus sayfayi yeniden yukler. */
const I18N = {
"Ana sayfa":"Home","Hesapla":"Explore","Karşılaştır":"Compare","Yöntem ve kanıt":"Method & evidence","Hakkında":"About",
"Bir yer seç":"Pick a place","Yöntemi incele":"See the method","Uyduların geçiş saati":"Satellite overpass time",
"Türkiye üzerinden gündüz geçişinin yerel güneş saati, yıllık medyan.":"Local solar time of the daytime overpass over Türkiye, yearly median.",
"Uydu":"Satellite","İlk yıl":"First year","Son yıl":"Last year",
"1999'de fırlatıldı. Yakıtı bitti, veri toplama Ocak 2027'de sona eriyor.":"Launched 1999. Out of fuel; data collection ends January 2027.",
"2002'de fırlatıldı. Yörüngesi kayıyor, 2027 sonbaharına kadar çalışacak.":"Launched 2002. Orbit drifting; operating until autumn 2027.",
"2011'de fırlatıldı. VIIRS taşıyor. FIRMS arşivinde verisi 2 Kasım 2026'ya kadar.":"Launched 2011. Carries VIIRS. FIRMS data ends 2 November 2026.",
"2017'de fırlatıldı. VIIRS taşıyor, geçiş saati sabit.":"Launched 2017. Carries VIIRS; overpass time is stable.",
"Terra ve Aqua yirmi yılı aşkın süredir Dünya'daki yangınları sayıyor. Yakıtları bitiyor ve geçiş saatleri kayıyor. Bu kaydı VIIRS'e kesintisiz devretmek için iki sensörün sayımını aynı ölçeğe çeviriyoruz. Bir yer seç, 2003'ten bugüne uyumlu yangın takvimini gör.":"Terra and Aqua have counted Earth's fires for more than twenty years. They are running out of fuel and their overpass times are drifting. To hand this record over to VIIRS without a break, we put both sensors' counts on the same scale. Pick a place and see its harmonized fire calendar from 2003 to today.",
"Bir ülke, il, ilçe ya da haritada çizdiğin bir alan için uyumlu yangın takvimi, yıllık seyir, kritik aylar ve sezon tablosu.":"Harmonized fire calendar, yearly course, critical months and season table for a country, province, district or an area you draw on the map.",
"İki yeri ya da aynı yerin iki dönemini yan yana koy: hangisinde sezon daha erken başlıyor, hangi aylar ağırlıkta.":"Put two places, or two periods of one place, side by side: where the season starts earlier, which months dominate.",
"Uyumun hangi ölçekte tuttuğu, 18 bölgedeki test sonuçları ve MODIS'in kayan saatinin kayda etkisi.":"At which scale the harmonization holds, test results in 18 regions, and how MODIS's drifting clock affects the record.",
"Bir yer seç, MODIS ve VIIRS'in birleştirildiği yangın takvimini gör.":"Pick a place and see the fire calendar with MODIS and VIIRS combined.",
"Örnek bölgeler":"Sample regions","Dünyada ara":"Search the world","Dosya yükle":"Upload files","Bölge":"Region",
"Türkiye ve çevresi":"Türkiye and surroundings","Yeni Güney Galler":"New South Wales","Haritada":"On the map",
"uyumlulaştırılmış":"harmonized","VIIRS ham":"VIIRS raw","Seçimi temizle":"Clear selection","CSV indir":"Download CSV",
"Dünya araması bu bilgisayardaki sunucuyla çalışır.":"World search runs with the server on this computer.",
"Proje klasöründe":"In the project folder, run","çalıştır; uygulama":"; the app opens at",
"adresinde açılır. Sunucu FIRMS anahtarını saklar, verileri hazırlar ve uyumlulaştırmayı yapar. Bu sayfa sunucusuz açıldığı için örnek bölgeler ve dosya yükleme kullanılabilir.":". The server keeps the FIRMS key, prepares the data and runs the harmonization. This page was opened without the server, so sample regions and file upload are available.",
"Ara":"Search","İndirilen bölgeler":"Downloaded regions",", anında hesaplanır":", computed instantly","yenile":"refresh",
"Son aramaların":"Recent searches","temizle":"clear","Haritaya tıklayınca seç":"Click on the map to select",
"ülke":"country","il":"province","ilçe":"district","kutu çiz":"draw a box","Tüm VIIRS uyduları":"All VIIRS satellites",
"Takvimi hesapla":"Compute calendar","Bir yer ara ya da haritada tıkla.":"Search for a place or click on the map.",
"FIRMS dosyalarını buraya bırak":"Drop FIRMS files here",
"Henüz bir yer seçmedin.":"No place selected yet.",
"Yukarıda bir yer ara, bir düğmeye bas ya da haritada tıkla; sonra Takvimi hesapla'ya bas.":"Search for a place above, press a button or click on the map; then press Compute calendar.",
"Haritada sürükleyerek bir alan seç; tıklarsan o noktanın çevresindeki 1°'lik alan seçilir. Hücreler 0,25° (yaklaşık 25 km): iki sensörün uyumunun geçerli olduğu en ince ölçek.":"Drag on the map to select an area; a click selects the 1° area around that point. Cells are 0.25° (about 25 km): the finest scale at which the two sensors agree.",
"MODIS-eşdeğeri tespit. En yoğun aylar":"MODIS-equivalent detections. Busiest months:",
"Güven: yüksek":"Confidence: high","Güven: orta":"Confidence: medium","Güven: düşük":"Confidence: low",
"yüksek":"high","orta":"medium","düşük":"low",
"Kalibrasyon yıllarında bol MODIS tespiti var ve yıllık toplamlar az sapıyor. Yıllık ve aylık sonuçlar güvenilir.":"Plenty of MODIS detections in the calibration years and small yearly deviations. Yearly and monthly results are reliable.",
"Tespit sayısı ya da yıllar arası sapma orta düzeyde. Eğilimler ve sezon güvenilir; tek tek yılları dikkatle yorumla.":"Detection count or year-to-year deviation is moderate. Trends and season are reliable; read single years with care.",
"Kalibrasyon için az MODIS tespiti var ya da yıllar arası sapma büyük. Sonuç yön gösterir; kesin sayı olarak kullanma.":"Few MODIS detections for calibration or large year-to-year deviation. Use the result as a direction, not an exact number.",
"Yanma takvimi (kesikli çizgi: VIIRS dönemi başlıyor)":"Burning calendar (dashed line: VIIRS era begins)",
"Yıllara göre":"By year","ortalama":"mean","rekor":"record","kırmızı":"red",": uyumlulaştırılmış ·":": harmonized ·",
"kahverengi":"brown",": gerçek MODIS ·":": actual MODIS ·","mavi kesikli":"blue dashed",
"Kritik aylar":"Critical months","Seçilen yılların ortalaması. Turuncu sütunlar ortalamanın üstündeki aylar.":"Mean of the selected years. Orange bars are months above the mean.",
"Sezon kayması":"Season shift","Ortalamadan sapma":"Deviation from mean","Sezon tablosu":"Season table","yıl":"year",
"ortalamaya göre":"vs. mean","sezon başı":"season start","tepe":"peak","sezon sonu":"season end","aktif gün":"active days",
"Sezon başı ve sonu, yıllık uyumlulaştırılmış aktivitenin %10'una ve %90'ına ulaşılan günler. Tepe, en yoğun 15 günlük pencerenin ortası.":"Season start and end are the days when 10% and 90% of the yearly harmonized activity is reached. Peak is the centre of the busiest 15-day window.",
"İki yeri ya da aynı yerin iki dönemini yan yana koy.":"Put two places, or two periods of one place, side by side.",
"İki yer":"Two places","İki dönem":"Two periods","Başka yer seç":"Pick another place","Yer seç":"Pick a place",
"Yeni yer seçmek için sunucu gerekir.":"Picking a new place needs the server.",
"çalıştır. Sunucusuz açılan sayfada yalnızca örnek bölgeler karşılaştırılabilir.":". Without the server only the sample regions can be compared.",
"Bu yeri kullan":"Use this place","Vazgeç":"Cancel","Bir yer ara, bir bölgeye bas ya da haritada tıkla.":"Search for a place, press a region or click on the map.",
"Ayların payı":"Share of months","Her ayın yıllık uyumlulaştırılmış aktiviteden aldığı pay. Büyüklükten bağımsız, mevsim şeklini karşılaştırır.":"Each month's share of the yearly harmonized activity. Independent of size; compares the shape of the season.",
"Sezon ilerleyişi":"Season progress","Yılın her gününe kadar biriken aktivitenin yüzdesi. Eğri ne kadar soldaysa sezon o kadar erken.":"Share of activity accumulated by each day of the year. The further left the curve, the earlier the season.",
"Yıllık uyumlulaştırılmış toplam (MODIS-eşdeğeri tespit).":"Yearly harmonized total (MODIS-equivalent detections).",
"Özet tablo":"Summary table","Dönem":"Period","Yıllık ortalama (MODIS-eşdeğeri)":"Yearly mean (MODIS-equivalent)",
"En yoğun ay":"Busiest month","Sezon başı (%10)":"Season start (10%)","Sezon ortası (%50)":"Season middle (50%)",
"Sezon sonu (%90)":"Season end (90%)","En yoğun 15 gün":"Busiest 15 days","Yıldan yıla değişkenlik":"Year-to-year variability",
"İki sensör aynı yangını farklı sayıyor. Nerede ve ne kadar birleştirilebildiklerini ölçtük.":"The two sensors count the same fire differently. We measured where, and how far, they can be combined.",
"18 bölgede test":"Tested in 18 regions",
"Her bölgede dönüşüm, Terra ve Aqua'nın yörüngesi henüz kaymamışken ilk yıllarda öğreniliyor ve sonraki yıllarda modelin hiç görmediği verilerle test ediliyor.":"In each region the transfer is learned in the early years, before Terra's and Aqua's orbits drifted, and tested on later years the model never saw.",
"Güven: kalibrasyon yıllarındaki MODIS tespit sayısı ve yıllık sapmaya göre. Az yanan bölgelerde güven düşük çıkıyor; bunu gizlemiyoruz.":"Confidence: based on the MODIS detection count in the calibration years and the yearly deviation. Regions with little fire get low confidence; we do not hide it.",
"VIIRS'i MODIS diline çevirmek":"Translating VIIRS into MODIS terms",
"Tek bir VIIRS uydusu (S-NPP), Terra ve Aqua'nın toplamından 4–5 kat fazla tespit yapıyor, çünkü pikseli 7 kat küçük. Ham sayıları üst üste koymak, 2012'deki sensör geçişinde takvimde yapay bir sıçrama yaratır. Bu yüzden her hücre-gündeki VIIRS sayısını MODIS'in göreceği sayıya çeviriyoruz.":"A single VIIRS satellite (S-NPP) makes 4–5 times more detections than Terra and Aqua combined, because its pixel is 7 times smaller. Stacking raw counts creates an artificial jump at the 2012 sensor change. So we convert the VIIRS count in every cell-day into the count MODIS would see.",
"Uyum hangi ölçekte tutuyor":"At which scale agreement holds",
"Karşılaştırma ölçeği büyüdükçe MODIS'in tespitlerinin yüzde kaçını VIIRS de görüyor. Tek piksel düzeyinde uyum düşük; literatürde iki uydu aynı anda geçtiğinde bile tarım alanlarında bu oran %47. Takvimi uyumun eşiği geçtiği ölçekte kuruyoruz.":"Share of MODIS detections that VIIRS also sees as the comparison scale grows. At single-pixel level agreement is low; even with simultaneous overpasses the literature reports 47% over croplands. We build the calendar at the scale where agreement passes the threshold.",
"Dolu sütun: hedefi geçen ölçek. Boş sütun: geçemeyen.":"Filled bar: scale that passes the target. Empty bar: does not.",
"Yangın, MODIS gelmeden sönüyor":"Fires die out before MODIS arrives",
"Bir olayın kaç ayrı VIIRS geçişinde görüldüğü, o yangının ne kadar sürdüğünü gösteriyor. Uzun süren yangını iki sensör de yakalıyor.":"The number of VIIRS overpasses that see an event shows how long the fire lasted. Long fires are caught by both sensors.",
"MODIS'in saati kayıyor":"MODIS's clock is drifting",
"Terra ve Aqua'nın yakıtı bitiyor ve yörüngeleri kayıyor. Güneş saati, uyduların Türkiye üzerinden her yıl geçtiği medyan yerel saati gösteriyor. Her halka bir yıl, içten dışa.":"Terra and Aqua are running out of fuel and their orbits are drifting. The sun dial shows the median local time at which each satellite passed over Türkiye each year. Each ring is one year, inside out.",
"Gölgeli dilim: gün içinde yangın aktivitesinin tepe yaptığı saatler (14:00–15:00).":"Shaded slice: hours of peak daily fire activity (14:00–15:00).",
"NASA Space Apps Challenge 2026, Ankara. Görev: Harmonization of MODIS and VIIRS Hot Spots.":"NASA Space Apps Challenge 2026, Ankara. Challenge: Harmonization of MODIS and VIIRS Hot Spots.",
"Neden":"Why","Kayıt el değiştiriyor":"The record changes hands",
"MODIS 2000'den beri küresel yangın kaydının omurgası. Terra'nın veri toplaması Ocak 2027'de, Aqua'nınki 2027 sonbaharında bitiyor. Kaydın devamı VIIRS'te, ama VIIRS'in pikseli 7 kat küçük ve aynı yangını çok daha fazla sayıyor.":"MODIS has been the backbone of the global fire record since 2000. Terra's data collection ends in January 2027, Aqua's in autumn 2027. The record continues with VIIRS, but its pixel is 7 times smaller and it counts the same fire many more times.",
"Ham birleştirme yanıltır":"Naive merging misleads",
"İki sensörün sayılarını yan yana koymak 2012'de yapay bir sıçrama, 2021'den sonra da yörünge kaymasından gelen sahte bir eğilim üretir. Ateş Takvimi her bölge için VIIRS'i MODIS'in göreceği sayıya çevirir ve bunun ne kadar tuttuğunu test eder.":"Putting the two sensors' counts side by side creates an artificial jump in 2012 and, after 2021, a false trend caused by orbit drift. Ateş Takvimi converts VIIRS into the count MODIS would see for each region and tests how well that holds.",
"Veri":"Data",
"MODIS Collection 6.1 (Terra ve Aqua, 1 km) ve VIIRS 375 m (S-NPP, NOAA-20, NOAA-21) aktif yangın tespitleri; arşiv (SP) ve yakın gerçek zamanlı (NRT) veri.":"MODIS Collection 6.1 (Terra and Aqua, 1 km) and VIIRS 375 m (S-NPP, NOAA-20, NOAA-21) active fire detections; archive (SP) and near-real-time (NRT) data.",
"Hazır veri setleri":"Prepared datasets","Harita ve arama":"Map and search",
"Yer arama ve sınırlar: OpenStreetMap Nominatim. Altlık harita: OpenStreetMap katkıcıları.":"Place search and boundaries: OpenStreetMap Nominatim. Base map: OpenStreetMap contributors.",
"Sınırlar":"Limitations","Tespit sayısı yanık alan değildir":"Detections are not burned area",
"Takvim, uydunun gördüğü sıcak nokta sayısını gösterir; yanan hektarı değil.":"The calendar shows the number of hot spots the satellite saw, not hectares burned.",
"Bulut ve geçiş saati":"Clouds and overpass time",
"Bulut altındaki yangınlar iki sensörde de görünmez. Bulut maskesi kullanılmadı.":"Fires under clouds are invisible to both sensors. No cloud mask was applied.",
"Ölçek":"Scale",
"Birleştirme 0,25° hücre ve gün ölçeğinde geçerli. Tek tek pikselleri eşleştirmek fiziksel olarak mümkün değil: iki uydu aynı anda geçtiğinde bile tarım alanlarında uyum %47.":"The harmonization holds at the 0.25° cell and day scale. Matching single pixels is physically impossible: even with simultaneous overpasses, agreement over croplands is 47%.",
"Kendi bilgisayarında":"On your own computer","Çalıştırma":"Running",". Uygulama":". The app opens at",
"adresinde açılır; dünya araması ve indirilen bölgeler bu modda çalışır.":"; world search and downloaded regions work in this mode.",
"Kaynaklar":"References",
"Veri: NASA FIRMS (MODIS C6.1, VIIRS 375 m). Harita: OpenStreetMap katkıcıları.":"Data: NASA FIRMS (MODIS C6.1, VIIRS 375 m). Map: OpenStreetMap contributors.",
"Ateş Takvimi, NASA Space Apps Challenge 2026 Ankara için hazırlandı.":"Ateş Takvimi (\"Fire Calendar\") was built for NASA Space Apps Challenge 2026 Ankara.",
"Sunucusuz açılan sayfada yalnızca örnek bölgeler var. Tüm bölgeler ve haritadan seçim için python sunucu.py ile aç.":"Without the server only the sample regions are available. Open with python sunucu.py for all regions and map selection.",
"Yıllar":"Years","Güven":"Confidence","İsabet":"Accuracy","r gün":"r day","r hafta":"r week","En yoğun yıl":"Busiest year",
"gerçek MODIS":"actual MODIS","çevrilmiş VIIRS":"converted VIIRS","VIIRS ham ÷ 10":"VIIRS raw ÷ 10",
"Piksel · ayak izi":"Pixel · footprint","±6 saat":"±6 hours","±24 saat":"±24 hours","Piksel · 2 km":"Pixel · 2 km",
"0.25° hücre":"0.25° cell","aynı gün":"same day","±1 gün":"±1 day","0.5° hücre":"0.5° cell","hedef %75":"target 75%",
"Mekânsal desen · 0.5° hücre":"Spatial pattern · 0.5° cell","Hücre × hafta · 0.5°":"Cell × week · 0.5°",
"Mekânsal desen · 1° hücre":"Spatial pattern · 1° cell","Hücre × hafta · 1°":"Cell × week · 1°","Bölgesel günlük seyir":"Regional daily course",
"1 geçiş":"1 overpass","2 geçiş":"2 overpasses","3 geçiş":"3 overpasses","4-5 geçiş":"4–5 overpasses",">=6 geçiş":"≥6 overpasses",
"yangın tepesi":"fire peak",
"Türkiye üzerinden gündüz geçişinin yerel güneş saati, yıllık medyan (FIRMS tespitlerinden). Boş daire ilk yıl, ok ucu son yıl. Kırmızı dilim gün içi yangın tepesi (14:00–15:00). VIIRS uyduları aynı saatte kalıyor.":"Local solar time of the daytime overpass over Türkiye, yearly median (from FIRMS detections). Open circle: first year; arrowhead: last year. Red slice: daily fire peak (14:00–15:00). VIIRS satellites keep the same time.",
"Terra'nın geçiş saati":"Terra's overpass time","Terra'nın gördüğü yangın, dünya geneli":"Fires seen by Terra, worldwide",
"Türkiye'de ham MODIS, uyumlu seriye göre":"Raw MODIS in Türkiye vs. harmonized series",
"Uyumlu seri VIIRS'e dayandığı için bu kaymadan etkilenmiyor.":"The harmonized series rests on VIIRS, so it is not affected by this drift.",
"Terra, Türkiye":"Terra, Türkiye","Aqua, dünya":"Aqua, world","Aqua, Türkiye":"Aqua, Türkiye","Terra, dünya":"Terra, world",
"Pantanal, Brezilya":"Pantanal, Brazil","Güney Kore":"South Korea","Kuzey Portekiz ve Galiçya":"Northern Portugal and Galicia",
"İsrail, Lübnan, Batı Suriye":"Israel, Lebanon, western Syria","Orta Şili: Valparaíso–Biobío":"Central Chile: Valparaíso–Biobío",
"İspanya: Huelva ve Sevilla":"Spain: Huelva and Seville","ABD: Oregon":"USA: Oregon","Kanada: Jasper":"Canada: Jasper",
"Yunanistan: Attika, Boiotia, Evia":"Greece: Attica, Boeotia, Euboea","Arnavutluk, Kuzey Makedonya, Karadağ":"Albania, North Macedonia, Montenegro",
"Havai: Maui":"Hawaii: Maui","Fransa: Gironde ve Landes":"France: Gironde and Landes",
"İspanya: Madrid, Ávila, Toledo, Guadalajara":"Spain: Madrid, Ávila, Toledo, Guadalajara","Kıbrıs":"Cyprus","Yunanistan: Girit":"Greece: Crete",
"Dünya (tüm ülkeler)":"World (all countries)","Seçilen alan":"Selected area","Çizilen kutu":"Drawn box",
"Hesapla · Ateş Takvimi":"Explore · Ateş Takvimi","Karşılaştır · Ateş Takvimi":"Compare · Ateş Takvimi",
"Yöntem ve kanıt · Ateş Takvimi":"Method & evidence · Ateş Takvimi","Hakkında · Ateş Takvimi":"About · Ateş Takvimi",
"Ateş Takvimi ana sayfa":"Ateş Takvimi home","Sayfalar":"Pages","Ne yapabilirsin":"What you can do","Veri kaynağı":"Data source",
"Dünya haritası: yalnızca uyduların gördüğü yangınlarla çizilmiş; üstünde Terra, Aqua ve S-NPP yörünge şeritleri":"World map drawn only from fires seen by satellites, with Terra, Aqua and S-NPP ground tracks",
"Ülke, il, ilçe ya da şehir yaz: Muğla, Manchester, Kaliforniya":"Type a country, province, district or city: Muğla, Manchester, California",
"Ülke, il, ilçe ya da şehir yaz: Antalya, Attika, Kaliforniya":"Type a country, province, district or city: Antalya, Attica, California",
"Yer ara":"Search place","Arama sonuçları":"Search results","İndirmesi yeni biten bölgeleri listeye ekle":"Add newly downloaded regions to the list",
"Seçim düzeyi":"Selection level","Karşılaştırma türü":"Comparison type","Karşılaştırmak için yer ara":"Search a place to compare",
"Uyduların yerel geçiş saati, ilk ve son yıl":"Local overpass time of the satellites, first and last year",
"Eğilim için en az 5 yıl seç.":"Select at least 5 years for a trend.",
"Seçilen dönemlerden birinde hiç tespit yok. Yıl aralığını değiştir.":"One of the selected periods has no detections. Change the year range.",
"Bu bölgede ve dönemde hiç yangın tespiti yok.":"No fire detections in this area and period.",
"veri parçaları indiriliyor":"downloading data parts",
"hesaplanıyor":"computing",
"Yer arama servisine ulaşılamadı (internet bağlantısını kontrol et).":"Could not reach the place search service (check your internet connection).",
"Bu işlem internet sürümünde yok.":"This action is not available in the online version.",
"Küre":"Globe","Düz harita":"Flat map","Toplam":"Total","Ay ay":"By month","Değişim":"Change","Oynat":"Play","Durdur":"Pause",
"Bu yerin takvimini hesapla":"Compute this place's calendar","Harita görünümü":"Map view","Küre görünümü":"Globe view","Ay":"Month",
"Dünya yangın küresi: uyduların gördüğü yangınlarla çizilmiş. Ok tuşlarıyla döndür, Enter ile ortadaki yeri seç.":"Fire globe drawn from fires seen by satellites. Rotate with the arrow keys, press Enter to select the centre.",
"Bunun için sunucu gerekir: python sunucu.py":"This needs the server: python sunucu.py",
"Rapor (PDF)":"Report (PDF)","Rapor indir (PDF)":"Download report (PDF)","Uydu izleri":"Satellite tracks","Bağlantıyı kopyala":"Copy link","Kopyalandı":"Copied","Kopyalanamadı":"Could not copy","Kendi sınırını yükle":"Upload your own boundary","GeoJSON ya da KML":"GeoJSON or KML",
"Veri: NASA FIRMS (MODIS C6.1, VIIRS 375 m). Yöntem: 0,25° hücre ve gün ölçeğinde VIIRS sayıları bölgeye özgü bir eğriyle MODIS eşdeğerine çevrilir; kalıcı ısı kaynakları ayıklanır. Sayılar yangın tespitidir, yanan alan değildir. Ateş Takvimi, NASA Space Apps Challenge 2026.":"Data: NASA FIRMS (MODIS C6.1, VIIRS 375 m). Method: at the 0.25° cell and day scale, VIIRS counts are converted to MODIS equivalents with a region-specific curve; persistent heat sources are removed. Numbers are fire detections, not burned area. Ateş Takvimi, NASA Space Apps Challenge 2026.",
"Bu bağlantıyı açmak için sunucu gerekir: python sunucu.py":"Opening this link needs the server: python sunucu.py",
"Dosya okunamadı: geçerli bir KML ya da GeoJSON değil.":"Could not read the file: not a valid KML or GeoJSON.",
"Dosyada alan (Polygon) bulunamadı. Nokta ve çizgiler kullanılamaz.":"No area (Polygon) found in the file. Points and lines cannot be used.",
"Koordinatlar enlem-boylam (WGS84) değil. Dosyayı EPSG:4326 olarak dışa aktar.":"Coordinates are not latitude-longitude (WGS84). Export the file as EPSG:4326.",
"Dosya çok büyük (en fazla 20 MB).":"File too large (20 MB at most).","Yüklenen sınır":"Uploaded boundary",
"Son günler":"Recent days","Şu an nerede yanıyor":"Where it is burning now","Yer":"Place","Tespit":"Detections","Normale göre":"vs. normal","takvimi aç":"open calendar","normalde az yanar":"rarely burns","yüklenen sınır":"uploaded boundary",
"Uydular":"Satellites","Döndür":"Rotate","Yörünge yılı":"Orbit year","Uydular ve dönüş":"Satellites and rotation","Gelişmiş":"Advanced",
"Turu başlat":"Start the tour","Turu kapat":"Close tour","Geri":"Back","İleri":"Next","Türkiye'nin takvimini aç":"Open Türkiye's calendar",
"Yirmi yıllık yangın kaydı":"A twenty-year fire record","Saat kayıyor, sayım da kayıyor":"The clock drifts, and so does the count","Kayıt VIIRS'e geçiyor":"The record moves to VIIRS","Sıra sende":"Your turn",
"MODIS 2000'den beri Dünya'daki yangınları sayıyor. Bu kürede kıyı çizgisi yok: haritayı yalnızca uyduların gördüğü yangınlar çiziyor.":"MODIS has counted Earth's fires since 2000. This globe has no coastlines: the map is drawn only by the fires satellites saw.",
"Uyduların yakıtı bitiyor, yörüngeleri kayıyor.":"The satellites are running out of fuel and their orbits are drifting.",
"Kürede bir yere tıkla ya da bir yer ara: o yerin 2003'ten bugüne uyumlu yangın takvimi saniyeler içinde açılır.":"Click a place on the globe or search for one: its harmonized fire calendar from 2003 to today opens in seconds.",
"Hesaplanıyor…":"Computing…","hazır veri setinde aranıyor":"looking in the prepared datasets"
};
const MON_S = {Oca:"Jan",Şub:"Feb",Mar:"Mar",Nis:"Apr",May:"May",Haz:"Jun",Tem:"Jul",Ağu:"Aug",Eyl:"Sep",Eki:"Oct",Kas:"Nov",Ara:"Dec"};
const MON_L = {Ocak:"January",Şubat:"February",Mart:"March",Nisan:"April",Mayıs:"May",Haziran:"June",Temmuz:"July",Ağustos:"August",Eylül:"September",Ekim:"October",Kasım:"November",Aralık:"December"};
const swapNum = t => t.replace(/(\d)([.,])(\d)/g, (m, a, s, b) => a + (s === "." ? "," : ".") + b);   /* tr 1.234,5 -> en 1,234.5 */
const tPct = t => t.replace(/%(\d+(?:[.,]\d+)?)/g, "$1%");
const tMon = t => t.replace(/(\d{1,2}) (Oca|Şub|Mar|Nis|May|Haz|Tem|Ağu|Eyl|Eki|Kas|Ara)(?![a-zçğıöşü])/g, (m, d, k) => d + " " + MON_S[k])
                   .replace(/(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)/g, m => MON_L[m]);
const N = s => swapNum(s);
/* kaliplar: [regex, cikti ya da fonksiyon]; metnin icinde aranir (sirayla) */
const I18R = [
  [/^Veri (.+) tarihine kadar$/, (m, a) => "Data through " + tMon(a)],
  [/Türkiye'de dönüşüm (\S+) yıllarından öğrenildi; modelin hiç görmediği (\S+) verisinde yıllık toplamı %(\d+) isabetle, günlük seyri r = ([\d,]+) ile tutturdu\./,
    (m, a, b, c, d) => `In Türkiye the transfer was learned from ${a}; on ${b} data the model never saw, it matched the yearly total with ${c}% accuracy and the daily course with r = ${N(d)}.`],
  [/Güveni yüksek (\d+) bölgede test isabeti %(\d+) ile %(\d+) arasında\./, "In the $1 high-confidence regions, test accuracy ranges from $2% to $3%."],
  [/2021'den sonra ham MODIS, uyumlu seriden her yıl daha fazla ayrılıyor: (\d+)(?:'de|'da|'te|'ta| yılında) %(\d+) fazla sayıyor, çünkü Aqua'nın geçişi öğleden sonraki yangın tepesine kaydı\./,
    "Since 2021 raw MODIS drifts further from the harmonized series every year: in $1 it over-counts by $2%, because Aqua's overpass moved into the afternoon fire peak."],
  [/^(.*), ([\d–]+): uyumlulaştırılmış toplam$/, (m, a, b) => { const w = a.includes("(tüm alan)"), c = a.replace(" (tüm alan)", "");
    return `${trx(c) || c}${w ? " (whole area)" : ""}, ${b}: harmonized total`; }],
  [/^\d{1,3}(\.\d{3})+$/, m => m.replace(/\./g, ",")],
  [/^(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık) ve (Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)$/, (m, a, b) => MON_L[a] + " and " + MON_L[b]],
  [/^\. Sezon tepesi yıllar arasında (.+)–(.+) aralığında gezdi\.$/, (m, a, b) => `. Across years the season peak ranged from ${tMon(a)} to ${tMon(b)}.`],
  [/Yıllık toplamda tipik sapma ±%(\d+)\./, "Typical deviation of the yearly total: ±$1%."],
  [/: bu alanda ve dönemde tespit yok\. Haritada başka bir alan seç ya da yıl aralığını genişlet\./, ": no detections in this area and period. Pick another area or widen the year range."],
  [/Kalibrasyon katsayısı k = ([\d,]+)\./, (m, a) => `Calibration coefficient k = ${N(a)}.`],
  [/Kalibrasyon yıllarında ([\d.]+) MODIS tespiti\./, (m, a) => `${N(a)} MODIS detections in the calibration years.`],
  [/Yıllık ortalama aktivite B tarafında \((.+?)\) A tarafına \((.+?)\) göre (.+?)\. /, (m, b, a, r) => `Yearly mean activity on side B (${trx(b) || b}) is ${tRel(r)} compared with side A (${trx(a) || a}). `],
  [/(.+?), (.+?) dönemi: yıllık ortalama aktivite (.+?) dönemine göre (.+?)\. /, (m, n, b, a, r) => `${trx(n) || n}, period ${b}: yearly mean activity is ${tRel(r)} compared with ${a}. `],
  [/Sezon iki tarafta da aşağı yukarı aynı günlerde başlıyor/, "The season starts at about the same time on both sides"],
  [/Sezon (.+?) tarafında (\d+) gün daha (erken|geç) başlıyor/, (m, n, d, e) => `The season starts ${d} days ${e === "erken" ? "earlier" : "later"} in ${trx(n) || n}`],
  [/ ve benzer uzunlukta sürüyor\. /, " and lasts about as long. "],
  [/ ve (\d+) gün daha (uzun|kısa) sürüyor\. /, (m, d, e) => ` and lasts ${d} days ${e === "uzun" ? "longer" : "shorter"}. `],
  [/En yoğun ay ikisinde de (\S+)\./, (m, a) => `The busiest month is ${MON_L[a] || a} for both.`],
  [/En yoğun ay (.+?) için (\S+), (.+?) için (\S+)\./, (m, a, ma, b, mb) => `Busiest month: ${MON_L[ma] || ma} for ${trx(a) || a}, ${MON_L[mb] || mb} for ${trx(b) || b}.`],
  [/^\(normalin ([\d,]+) katı\)$/, (m, a) => `(${N(a)}× normal)`],
  [/^eğitim (\S+), test (\S+)$/, "training $1, test $2"],
  [/^([\d,]+) kat$/, (m, a) => N(a) + "×"],
  [/^([+−-]?\d+) gün$/, "$1 days"],
  [/VIIRS'in gördüğü ([\d.]+) yangın olayının %(\d+)'[ıiuü] en fazla iki geçiş \(~1 saatten kısa\) sürüyor\./, (m, a, b) => `Of the ${N(a)} fire events VIIRS saw, ${b}% last at most two overpasses (under ~1 hour).`],
  [/Bunlar büyük çoğunlukla anız yangınları ve MODIS geçişine kadar sönüyorlar\./, "Most are stubble fires that die out before MODIS passes."],
  [/Bu yüzden tek piksel düzeyinde düşük uyum bir hata değil, bölgedeki yangınların kısa ömrünün sonucu\./, "So low pixel-level agreement is not an error; it follows from the short life of the region's fires."],
  [/^([\d.]+) olay$/, (m, a) => N(a) + " events"],
  [/Kalın kırmızı çizgi çevrilmiş VIIRS, kahverengi çizgi gerçek MODIS\./, "Thick red line: converted VIIRS; brown line: actual MODIS."],
  [/İki çizgi aynı günlerde yükselip alçalıyor \(r = ([\d,]+)\)\./, (m, a) => `Both lines rise and fall on the same days (r = ${N(a)}).`],
  [/MODIS tespitlerinin %(\d+)'[a-zçğıöşü]+ VIIRS'in o gün boş gördüğü hücrelerde, yani iki geçiş arasında başlayıp sönen yangınlarda\./, "$1% of MODIS detections fall in cells VIIRS saw empty that day, i.e. fires that started and died between two overpasses."],
  [/Bu kısım (.+?) öncesinde öğrenilen tek bir katsayıyla telafi ediliyor \(k = ([\d,]+)\)\./, (m, a, b) => `This part is compensated by a single coefficient learned before ${tMon(a)} (k = ${N(b)}).`],
  [/^(\d{1,2}) (Oca|Şub|Mar|Nis|May|Haz|Tem|Ağu|Eyl|Eki|Kas|Ara)$/, (m, d, k) => d + " " + MON_S[k]],
  [/^(Oca|Şub|Mar|Nis|May|Haz|Tem|Ağu|Eyl|Eki|Kas|Ara)$/, m => MON_S[m]],
  [/^(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)$/, m => MON_L[m]],
  [/^(\d+) → (\d+) arasında (\d+) dakika sabaha kaydı\. Yangınlar henüz başlamadan geçiyor\.$/, "Moved $3 minutes earlier between $1 and $2. It now passes before fires get going."],
  [/^S-NPP'nin her 1\.000 tespitine karşı: (\d+) yılında (\d+), (\d+) yılında (\d+)\.$/, "Per 1,000 S-NPP detections: $2 in $1, $4 in $3."],
  [/^(\d+) yılında olduğundan fazla sayıyor: Aqua öğleden sonraki yangın tepesine kaydı\.$/, "Over-counts in $1: Aqua moved into the afternoon fire peak."],
  [/^([\d:]+) · sabit$/, "$1 · stable"],
  [/Terra ve Aqua'nın yakıtı bitiyor ve yörüngeleri kayıyor\. Güneş saati, uyduların Türkiye üzerinden geçtiği yıllık medyan yerel saati gösteriyor\. Her halka bir yıl, içten dışa (\d+)'[a-zçğıöşü]+ (\d+)'[a-zçğıöşü]+\./,
    "Terra and Aqua are running out of fuel and their orbits are drifting. The sun dial shows the yearly median local time at which each satellite passed over Türkiye. Each ring is one year, from $1 (inside) to $2 (outside)."],
  [/Yıllık MODIS tespit sayısının S-NPP'ye oranı, kalıcı kaynaklar çıkarıldıktan sonra\./, "Ratio of yearly MODIS detections to S-NPP, after removing persistent sources."],
  [/Türkiye'de 2021'den sonra iki MODIS uydusunun oranı da yükseliyor; ham MODIS bu yüzden uyumlu seriden giderek ayrılıyor\./, "In Türkiye both MODIS ratios rise after 2021, so raw MODIS drifts away from the harmonized series."],
  [/Kesikli çizgiler dünyanın tamamı: Terra'nın oranı (\d+)'[a-zçğıöşü]+ (\d+)'[a-zçğıöşü]+ %(\d+) düştü, çünkü geçişi sabaha, yangınlar başlamadan önceki saatlere kaydı\./, "Dashed lines: the whole world. Terra's ratio fell $3% from $1 to $2 because its overpass moved to the morning, before fires get going."],
  [/Çubuk: sezonun %10–%90 aralığı, nokta: en yoğun 15 gün\./, "Bar: 10–90% of the season; dot: busiest 15 days."],
  [/ Kesikli çizgi tepenin eğilimi: /, " Dashed line: trend of the peak: "],
  [/tepe belirgin kaymadı/, "the peak did not shift clearly"], [/sezon başı belirgin kaymadı/, "the season start did not shift clearly"],
  [/(tepe|sezon başı) her 10 yılda yaklaşık (\d+) gün (erkene|geçe) kaydı \(r = ([\d,\-−]+)\)/, (m, w, d, e, r) => `${w === "tepe" ? "the peak" : "the season start"} moved about ${d} days ${e === "erkene" ? "earlier" : "later"} per decade (r = ${N(r)})`],
  [/Her yılın uyumlulaştırılmış toplamı, seçilen yılların ortalamasına göre\./, "Each year's harmonized total relative to the mean of the selected years."],
  [/En yoğun yıl (\d+) \(([+−-])%(\d+)\), en sakin yıl (\d+) \(([+−-])%(\d+)\)\./, "Busiest year $1 ($2$3%), calmest year $4 ($5$6%)."],
  [/: VIIRS ham ÷ 10\. Gölge: %80 belirsizlik aralığı \(kalibrasyon yıllarındaki yıllık sapmadan, ±%(\d+)\)\./, ": VIIRS raw ÷ 10. Shading: 80% uncertainty band (from the yearly deviation in the calibration years, ±$1%)."],
  [/İki sensörün birlikte olduğu yıllarda uyumlulaştırılmış toplam, gerçek MODIS'in ([\d,]+) katı\./, (m, a) => `In years with both sensors the harmonized total is ${N(a)}× actual MODIS.`],
  [/^Haritayı (\d+)–(\d+) arasında VIIRS'in gördüğü yangınlar çiziyor\. Şeritler: gündüz geçişleri, gerçek tarama genişliğinde\.$/, "The map is drawn by fires VIIRS saw in $1–$2. Bands: daytime overpasses at true swath width."],
  [/Hazır veri setinden anında hesaplandı \((.+?) veri seti (\S+), MODIS \+ VIIRS, kalıcı kaynaklar ayıklanmış\)\. Kullanılan: (.+?), (\d+) hücre\./,
    (m, a, y, rule, n) => `Computed instantly from a prepared dataset (${trx(a) || a}, ${y}, MODIS + VIIRS, persistent sources removed). Used: ${rule.replace("alanının en az yarısı sınırın içinde kalan 0,25° hücreler", "0.25° cells with at least half their area inside the boundary").replace("bölge bir hücreden küçük; en çok örtüşen 0,25° hücre", "area smaller than one cell; best-overlapping 0.25° cell")}, ${n} cells.`],
  [/Bu bölge hazır veri setlerinin dışında: FIRMS'ten canlı indirilecek, ≈ ([\d.]+) istek \((\S+)\)\. Daha önce indirilmiş pencereler anında gelir; yeni bir bölgede yıl başına yaklaşık ½–1 dakika\./,
    (m, n, y) => `This area is outside the prepared datasets: it will be downloaded live from FIRMS, ≈ ${N(n)} requests (${y}). Windows downloaded before arrive instantly; a new area takes about ½–1 minute per year.`],
  [/^Dünya veri seti (\S+) yıllarını kapsıyor; yıl aralığı buna göre ayarlandı\. Hesap anında gelir\.$/, "The world dataset covers $1; the year range was adjusted. Results arrive instantly."],
  [/^Dünya veri setinden \((\S+)\): hesap anında gelir\.$/, "From the world dataset ($1): results arrive instantly."],
  [/^Bu bölge "(.+)" hazır veri setinin içinde \((\S+)\): hesap anında gelir\.$/, (m, a, y) => `This area is inside the prepared dataset "${trx(a) || a}" (${y}): results arrive instantly.`],
  [/^Kutu çok küçük; sürükleyerek daha geniş bir alan çiz\.$/, "Box too small; drag to draw a wider area."],
  [/^(ülke|il|ilçe) sınırı aranıyor…$/, (m, a) => `looking up ${({ "ülke": "country", "il": "province", "ilçe": "district" })[a]} boundary…`],
  [/^sınır aranıyor…$/, "looking up boundary…"], [/^aranıyor…$/, "searching…"], [/^hesaplanıyor…$/, "computing…"],
  [/^"(.+)" için sonuç yok\. Başka yazımla dene.*$/, "No results for \"$1\". Try another spelling."],
  [/^(\d+) sonuç\. Birini seç\.$/, "$1 results. Pick one."],
  [/(.+)\. Sonuçlar aşağıda; bölge listesine de eklendi\.$/, (m, a) => `${trx(a) || a}. Results below; also added to the region list.`],
  [/^([\d.]+) hücre-gün hazır dünya verisinden \(([\d.,]+) sn\)/, (m, a, b) => `${N(a)} cell-days from the world dataset (${N(b)} s)`],
  [/Hazır dünya verisinden tarayıcıda hesaplandı \(Dünya veri seti (\S+), MODIS \+ VIIRS, kalıcı kaynaklar ayıklanmış\)\. Kullanılan: (.+?), (\d+) hücre\.( İnternet sürümünde veri (\S+) arası; dönem buna göre kırpıldı\.)?/,
    (m, y, rule, n, c, cy) => `Computed in your browser from the world dataset (${y}, MODIS + VIIRS, persistent sources removed). Used: ${rule.replace("alanının en az yarısı sınırın içinde kalan 0,25° hücreler", "0.25° cells with at least half their area inside the boundary").replace("bölge bir hücreden küçük; en çok örtüşen 0,25° hücre", "area smaller than one cell; best-overlapping 0.25° cell")}, ${n} cells.` + (c ? ` The online version has data for ${cy}; the period was trimmed.` : "")],
  [/^Bölge internet sürümü için çok geniş \((\d+)° × (\d+)°\)\. Daha küçük bir alan seç ya da bilgisayardaki tam sürümü kullan\.$/, "Area too large for the online version ($1° × $2°). Pick a smaller area or use the full desktop version."],
  [/^İnternet sürümünde yalnızca (\S+) yılları var\.$/, "The online version only covers $1."],
  [/^Veri parçası indirilemedi \((\S+), (\d+)\)\.$/, "Could not download data part ($1, $2)."],
  [/^Yer arama servisi hata verdi \((\d+)\)\. Biraz sonra tekrar dene\.$/, "The place search service returned an error ($1). Try again shortly."],
  [/^(-?[\d.]+), (-?[\d.]+) çevresi$/, "around $1, $2"],
  [/^([\d.]+) hücre-gün hazır veri setinden \(([\d.,]+) sn\)/, (m, a, b) => `${N(a)} cell-days from the prepared dataset (${N(b)} s)`],
  [/^([\d.]+) hücre-gün hazır \((\d+) sn\)/, (m, a, b) => `${N(a)} cell-days ready (${b} s)`],
  [/^FIRMS: (\d+)\/(\d+) pencere \((\d+) önbellekten\)/, "FIRMS: $1/$2 windows ($3 from cache)"],
  [/^FIRMS'ten (\d+) pencere alınıyor/, "Fetching $1 windows from FIRMS"],
  [/^(\d+) işleniyor$/, "processing $1"],
  [/^firms\.modaps\.eosdis\.nasa\.gov\/download adresinden.*$/, "Select the MODIS and VIIRS CSV files you downloaded from firms.modaps.eosdis.nasa.gov/download for the same area and period (archive or NRT). You can select several files. Everything is computed in your browser; files are not sent anywhere."],
  [/^Küreyi (\d+)–(\d+) arasında VIIRS'in gördüğü yangınlar çiziyor\. Sürükleyerek döndür, bir yere tıklayarak seç\./, "The globe is drawn by fires VIIRS saw in $1–$2. Drag to rotate, click a place to select it."],
  [/^Ay ay yangın: (\d+)–(\d+) toplamı\. Yangın mevsiminin yıl içinde dünyayı dolaşmasını izle\./, "Fire by month, $1–$2 total. Watch the fire season travel around the world through the year."],
  [/^Mavi: yangın azaldı, kırmızı: arttı\. (\d+)–(\d+) dönemi, (\d+)–(\d+) dönemine göre \(aynı sensör: VIIRS S-NPP\)\./, "Blue: less fire, red: more. $1–$2 compared with $3–$4 (same sensor: VIIRS S-NPP)."],
  [/^([\d,]+)° ([KG]), ([\d,]+)° ([DB])$/, (m, a, b, c, d) => `${N(a)}° ${b === "K" ? "N" : "S"}, ${N(c)}° ${d === "D" ? "E" : "W"}`],
  [/^Ateş Takvimi yangın raporu\. Dönem (\S+)\. Hazırlanma tarihi (.+)\.$/, (m, a, b) => `Ateş Takvimi fire report. Period ${a}. Prepared on ${tMon(b)}.`],
  [/^Sınır yüklendi: (.+): (\d+) alan, ([\d.]+) nokta\. Yılları seç, sonra Takvimi hesapla'ya bas\.$/, (m, f, a, n) => `Boundary loaded: ${f}: ${a} areas, ${N(n)} points. Pick the years, then press Compute calendar.`],
  [/^(.+): son (\d+) günde uyduların gördüğü yangınlar \(VIIRS NOAA-20 ve NOAA-21\)\./, (m, a, n) => `${tMon(a)}: fires seen by satellites in the last ${n} days (VIIRS NOAA-20 and NOAA-21).`],
  [/^(.+) arasında VIIRS'in \(NOAA-20 ve NOAA-21\) en çok yangın gördüğü yerler\.(?: Normal: aynı günlerin (\d+)–(\d+) ortalaması \(VIIRS S-NPP\)\.)? Sayılar 3°×3°'lik alan içindir\.$/,
    (m, a, y0, y1) => `Places where VIIRS (NOAA-20 and NOAA-21) saw the most fire, ${tMon(a)}.` + (y0 ? ` Normal: the ${y0}–${y1} mean of the same days (VIIRS S-NPP).` : "") + " Counts are for a 3°×3° area."],
  [/^Terra'nın geçişi ([\d:]+) iken ([\d:]+) oldu; Aqua'nınki ([\d:]+) iken ([\d:]+)\. Kesikli halka (\d+) yörüngesi\./, "Terra's overpass moved from $1 to $2; Aqua's from $3 to $4. The dashed ring is the $5 orbit."],
  [/ Terra artık yangınlar başlamadan geçiyor: dünya genelinde gördüğü yangın %(\d+) azaldı\./, " Terra now passes before fires get going: worldwide it sees $1% fewer fires."],
  [/^Terra 2027 başında, Aqua 2027 sonbaharında kapanıyor\. Kaydı VIIRS sürdürecek, ama aynı yangını 4–5 kat fazla sayıyor\. İki sayımı aynı ölçeğe çeviriyoruz(?:: modelin görmediği yıllarda Türkiye'nin yıllık toplamını %(\d+) isabetle tutturuyor)?\.$/,
    (m, x) => "Terra shuts down in early 2027, Aqua in autumn 2027. VIIRS will carry the record on, but it counts the same fire 4–5 times more. We put both counts on one scale" + (x ? `: on years the model never saw, it recovers ${x}% of Türkiye's yearly total.` : ".")],
  [/ Halkalar: uyduların (\d+) yörüngeleri, gerçek yükseklikte; kesikli halka (\d+) yörüngesi\. Gölge: gece\.$/, " Rings: the satellites' $1 orbits at true altitude; the dashed ring is the $2 orbit. Shade: night."],
  [/Seçilen nokta çevresi \((.+?) boyutunda\)/, "Area around the selected point"],
  [/İnternet yok ve bu yer daha önce aranmamış\..*/, "No internet and this place was not searched before. Click on the map, draw a box or type \"lat, lon\" (e.g. 38.4, 27.1)."],
];
/* metnin icinde gecen sabit ifadeler (kalibrasyon notu vb.) */
const I18P = [
  ["Kalibrasyon yıllarında bol MODIS tespiti var ve yıllık toplamlar az sapıyor. Yıllık ve aylık sonuçlar güvenilir.", "Plenty of MODIS detections in the calibration years and small yearly deviations. Yearly and monthly results are reliable."],
  ["Tespit sayısı ya da yıllar arası sapma orta düzeyde. Eğilimler ve sezon güvenilir; tek tek yılları dikkatle yorumla.", "Detection count or year-to-year deviation is moderate. Trends and season are reliable; read single years with care."],
  ["Kalibrasyon için az MODIS tespiti var ya da yıllar arası sapma büyük. Sonuç yön gösterir; kesin sayı olarak kullanma.", "Few MODIS detections for calibration or large year-to-year deviation. Use the result as a direction, not an exact number."],
  [" ve ark.", " et al."], ["Sofiev ve Kouznetsov", "Sofiev and Kouznetsov"],
  ["bölgeye özgü çok yıllı eğri", "region-specific multi-year curve"], ["Türkiye çok yıllı eğrisi (bölgeye özgü eğri için tespit az)", "Türkiye multi-year curve (too few detections for a regional curve)"],
  ["test döneminde toplam isabet", "total accuracy in test period"], ["test günlük seyir", "test daily course"],
  ["yörünge kayması yıllarında ham MODIS ÷ uyumlu seri", "in orbit-drift years raw MODIS ÷ harmonized series"], ["eğitim", "training"],
  ["S-NPP 2 Kasım 2026'da kapandığı için o günden sonrası NOAA-20/21 ortalamasıyla hesaplandı.", "S-NPP ended on 2 November 2026; later days use the NOAA-20/21 mean."],
];
function tRel(r) {
  if (r === "neredeyse aynı") return "nearly the same";
  let m = r.match(/^%(\d+) daha (yüksek|düşük)$/); if (m) return `${m[1]}% ${m[2] === "yüksek" ? "higher" : "lower"}`;
  m = r.match(/^([\d,]+) kat$/); if (m) return `${N(m[1])}×`;
  return r;
}
function trx(t) {
  const k = t.replace(/\s+/g, " ").trim();
  if (!k) return null;
  if (I18N[k]) return I18N[k];
  let out = k, hit = false;
  for (const [re, rep] of I18R) { if (re.test(out)) { out = out.replace(re, rep); hit = true; } }
  for (const [a, b] of I18P) if (out.includes(a)) { out = out.split(a).join(b); hit = true; }
  if (hit) out = tPct(tMon(out));
  return hit ? out : null;
}
let LANG = "tr";
try { LANG = localStorage.getItem("ates_lang") === "en" ? "en" : "tr"; } catch (e) {}
if (/[?&]lang=en\b/.test(location.search)) LANG = "en"; else if (/[?&]lang=tr\b/.test(location.search)) LANG = "tr";
const I18_ATTR = ["placeholder", "title", "aria-label", "data-tip"];
function i18Node(n) {
  if (n.nodeType === 3) {
    if (n.__en !== undefined && n.nodeValue === n.__en) return;
    const p = n.parentElement; if (!p || p.closest("script,style,.noi18n")) return;
    const v = n.nodeValue, t = trx(v);
    if (t != null) { const lead = v.match(/^\s*/)[0], tail = v.match(/\s*$/)[0]; n.__en = lead + t + tail; n.nodeValue = n.__en; }
  } else if (n.nodeType === 1) {
    if (n.closest && n.closest("script,style,.noi18n")) return;
    for (const a of I18_ATTR) { const v = n.getAttribute && n.getAttribute(a); if (v && n["__en_" + a] !== v) { const t = trx(v); if (t != null) { n["__en_" + a] = t; n.setAttribute(a, t); } } }
    const w = document.createTreeWalker(n, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
    let c; while ((c = w.nextNode())) { if (c.nodeType === 3) i18Node(c); else for (const a of I18_ATTR) { const v = c.getAttribute(a); if (v && c["__en_" + a] !== v) { const t = trx(v); if (t != null) { c["__en_" + a] = t; c.setAttribute(a, t); } } } }
  }
}
function i18Title() { const t = trx(document.title); if (t) document.title = t; }
function i18Start() {
  document.documentElement.lang = "en"; document.querySelector("main") && document.querySelector("main").setAttribute("lang", "en");
  i18Node(document.body); i18Title();
  new MutationObserver(ms => { for (const m of ms) {
      if (m.type === "characterData") i18Node(m.target);
      else if (m.type === "attributes") { const el = m.target, a = m.attributeName, v = el.getAttribute(a); if (v && el["__en_" + a] !== v) { const t = trx(v); if (t != null) { el["__en_" + a] = t; el.setAttribute(a, t); } } }
      else m.addedNodes.forEach(i18Node); } i18Title(); })
    .observe(document.body, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: I18_ATTR });
  new MutationObserver(i18Title).observe(document.querySelector("title"), { childList: true });
}
(function langButton() {
  const b = document.getElementById("langbtn"); if (!b) return;
  b.textContent = LANG === "en" ? "TR" : "EN";
  b.setAttribute("aria-label", LANG === "en" ? "Türkçe" : "English");
  b.addEventListener("click", () => { const nx = LANG === "en" ? "tr" : "en"; try { localStorage.setItem("ates_lang", nx); } catch (e) {}
    location.href = location.pathname + "?lang=" + nx + location.hash; });
})();
if (LANG === "en") i18Start();
</script>
"""

if __name__ == "__main__":
    main()
