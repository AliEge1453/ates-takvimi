"""
Ates Takvimi - yerel sunucu (dunya geneli arama)
================================================
Uygulamayi tarayicida acar ve dunyanin herhangi bir yeri icin MODIS-VIIRS uyumlu yanma takvimi hesaplar.

    cd C:\\nasa_harmonization
    python sunucu.py                 ->  tarayicida http://localhost:8765

Ne yapar:
  * /api/ara?q=Manchester          OpenStreetMap Nominatim ile yer arar (ulke / il / ilce sinirlari poligon olarak)
  * /api/ters?lat=..&lon=..&duzey=  haritada tiklanan noktanin ulke / il / ilce sinirini bulur
  * /api/bolge (POST)              secilen sinir + yil araligi icin FIRMS'ten MODIS ve VIIRS'i indirir (paralel,
                                   data_cache/ onbellegini kullanir), sinirin icine kirpar, kalici kaynaklari ayiklar,
                                   0.25 derece hucre-gun tablosu kurar ve bolgeye ozgu donusum egrisiyle uyumlulastirir
  * /api/is?id=.. , /api/sonuc?id=..   isin ilerlemesi ve sonucu

MAP_KEY map_key.txt'den okunur ve tarayiciya hic gonderilmez. Sunucu sadece bu bilgisayardan (127.0.0.1) erisilebilir.

Bayraklar:  --port 8765   --paralel 6   --cevrimdisi (internete cikmaz, sadece onbellek)
"""

import gzip
import hashlib
import json
import re
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import requests

import harmonize_v12 as h
import build_app as ba

HERE = Path(__file__).resolve().parent
PORT = int(h._argval("--port", 8765))
WORKERS = int(h._argval("--paralel", 6))
OFFLINE = "--cevrimdisi" in sys.argv
CELL = 0.25
UA = {"User-Agent": "AtesTakvimi/1.0 (NASA Space Apps 2026 Ankara takim projesi; yerel kullanim)"}
GEO_DIR = h.CACHE_DIR / "geo"
FIRMS = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
MAX_SPAN_DEG = 60          # bundan genis kutular reddedilir (FIRMS kotasi ve bellek)
LEVEL_ZOOM = {"ulke": 3, "il": 5, "ilce": 8, "sehir": 10}
LEVEL_NAME = {"ulke": "ülke", "il": "il / eyalet", "ilce": "ilçe", "sehir": "şehir"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------
# 1. FIRMS: onbellek dizini + pencere indirme
# ----------------------------------------------------------------------------
CACHE_RE = re.compile(r"^(MODIS_(?:SP|NRT)|VIIRS_(?:SNPP|NOAA20|NOAA21)_(?:SP|NRT))_(-?[\d.]+)_(-?[\d.]+)_(-?[\d.]+)_(-?[\d.]+)_"
                      r"(\d{8})_(\d+)d_asof(\w+)\.csv$")


class CacheIndex:
    """data_cache/ icindeki pencere dosyalarini (kaynak, baslangic, gun) -> [(kutu, asof, yol)] olarak tutar.
    Istenen kutu daha once indirilmis bir kutunun icindeyse o dosya okunup kirpilir (yeniden indirme yok)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.idx = {}
        h.CACHE_DIR.mkdir(exist_ok=True)
        for p in h.CACHE_DIR.glob("*.csv"):
            self._add(p)

    def _add(self, p):
        m = CACHE_RE.match(p.name)
        if not m:
            return
        src, w, s, e, n, st, nd, asof = m.groups()
        key = (src, st, int(nd))
        self.idx.setdefault(key, []).append(((float(w), float(s), float(e), float(n)), asof, p))

    def add(self, p):
        with self.lock:
            self._add(p)

    def find(self, src, start, days, bbox):
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        w, s, e, n = bbox
        best = None
        with self.lock:
            for (W, S, E, N), asof, p in self.idx.get((src, f"{start:%Y%m%d}", days), []):
                if asof not in ("sp", today):
                    continue
                if W <= w + 1e-9 and S <= s + 1e-9 and E >= e - 1e-9 and N >= n - 1e-9:
                    area = (E - W) * (N - S)
                    if best is None or area < best[0]:
                        best = (area, p)
        return best[1] if best else None


CACHE = None


def bbox_str(b):
    return ",".join(f"{v:g}" for v in b)


def read_csv_safe(p):
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def clip_bbox(df, bbox):
    if df.empty or "latitude" not in df:
        return df
    w, s, e, n = bbox
    m = (df["longitude"] >= w) & (df["longitude"] <= e) & (df["latitude"] >= s) & (df["latitude"] <= n)
    return df[m]


def fetch_window(key, src, bbox, start, days):
    """Tek 5 gunluk pencere. Dondurur (DataFrame, onbellekten_mi)."""
    hit = CACHE.find(src, start, days, bbox)
    if hit is not None:
        return clip_bbox(read_csv_safe(hit), bbox), True
    if OFFLINE:
        return pd.DataFrame(), True
    asof = "sp" if src.endswith("_SP") else datetime.now(timezone.utc).strftime("%Y%m%d")
    cp = h.CACHE_DIR / f"{src}_{bbox_str(bbox).replace(',', '_')}_{start:%Y%m%d}_{days}d_asof{asof}.csv"
    url = f"{FIRMS}/{key}/{src}/{bbox_str(bbox)}/{days}/{start:%Y-%m-%d}"
    waits, last = 0, None
    for attempt in range(1, 6):
        try:
            h._quota_guard(key)
            r = requests.get(url, timeout=120)
            if r.status_code == 400:
                body = (r.text or "").strip()
                if "invalid" in body.lower() and "key" in body.lower():
                    raise RuntimeError("FIRMS MAP_KEY geçersiz (map_key.txt)")
                if waits < 15:
                    waits += 1
                    h._wait_for_quota(key, body)
                    continue
            r.raise_for_status()
            text = r.text.strip()
            if "<html" in text[:200].lower() or text.lower().startswith("invalid"):
                raise RuntimeError(f"FIRMS hatası ({src} {start}): {text[:120]}")
            df = pd.read_csv(StringIO(text)) if text else pd.DataFrame()
            df.to_csv(cp, index=False)
            CACHE.add(cp)
            return df, False
        except RuntimeError:
            raise
        except Exception as e:                       # baglanti / sunucu hatasi
            last = e
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"FIRMS'e ulaşılamadı ({src} {start}): {last}")


# ----------------------------------------------------------------------------
# 2. Geometri: GeoJSON -> halkalar, kutu, nokta-icinde
# ----------------------------------------------------------------------------
def rings_of(geom):
    t, c = geom.get("type"), geom.get("coordinates")
    polys = [c] if t == "Polygon" else (c if t == "MultiPolygon" else [])
    out = []
    for poly in polys:
        for ring in poly:
            a = np.asarray(ring, float)
            if len(a) >= 3:
                if not np.allclose(a[0], a[-1]):
                    a = np.vstack([a, a[:1]])
                out.append(a)
    return out


def geom_bbox(rings):
    a = np.vstack(rings)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def snap_bbox(b):
    w, s, e, n = b
    f = lambda v: np.floor(v / CELL) * CELL
    c = lambda v: np.ceil(v / CELL) * CELL
    return (max(-180.0, f(w)), max(-90.0, f(s)), min(180.0, c(e)), min(90.0, c(n)))


def points_in(px, py, rings):
    """Cift-tek kurali (delikler ve coklu poligonlar dahil). Noktalar enleme gore siralanir; her kenar sadece
    kendi enlem araligindaki noktalara bakar, bu yuzden ulke boyutunda sinirlarda da hizli."""
    n = len(px)
    if n == 0:
        return np.zeros(0, bool)
    order = np.argsort(py, kind="stable")
    ys, xs = py[order], px[order]
    par = np.zeros(n, bool)
    for ring in rings:
        x0, y0, x1, y1 = ring[:-1, 0], ring[:-1, 1], ring[1:, 0], ring[1:, 1]
        lo, hi = np.minimum(y0, y1), np.maximum(y0, y1)
        i0s, i1s = np.searchsorted(ys, lo, "left"), np.searchsorted(ys, hi, "left")
        for k in np.nonzero((i1s > i0s) & (y0 != y1))[0]:
            i0, i1 = i0s[k], i1s[k]
            yy = ys[i0:i1]
            xint = x0[k] + (yy - y0[k]) * (x1[k] - x0[k]) / (y1[k] - y0[k])
            par[i0:i1] ^= xs[i0:i1] < xint
    out = np.zeros(n, bool)
    out[order] = par
    return out


def bbox_geom(w, s, e, n):
    return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


# ----------------------------------------------------------------------------
# 3. Yer arama (OpenStreetMap Nominatim, saniyede en fazla 1 istek, disk onbellegi)
# ----------------------------------------------------------------------------
NOM_LOCK = threading.Lock()
NOM_LAST = [0.0]


def nominatim(path, params):
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1((path + json.dumps(params, sort_keys=True)).encode("utf-8")).hexdigest()[:24]
    cp = GEO_DIR / f"{key}.json"
    if cp.exists():
        return json.loads(cp.read_text("utf-8"))
    if OFFLINE:
        raise NoNet("çevrimdışı mod")
    with NOM_LOCK:
        wait = 1.1 - (time.time() - NOM_LAST[0])
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(f"https://nominatim.openstreetmap.org/{path}", params=params, headers=UA, timeout=8)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise NoNet(str(e)) from e
        finally:
            NOM_LAST[0] = time.time()
    r.raise_for_status()
    data = r.json()
    cp.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    return data


class NoNet(RuntimeError):
    """Yer arama servisine ulasilamadi (internet yok)."""


TR_LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")


def norm(t):
    return (t or "").translate(TR_LOWER).lower().strip()


def offline_search(q):
    """Internet yokken arama: (1) "enlem, boylam" (2) hazir bolgeler (3) daha once aranan/tiklanan yerler (onbellek)."""
    out = []
    m = re.match(r"^\s*(-?\d+(?:[.,]\d+)?)\s*[,; ]\s*(-?\d+(?:[.,]\d+)?)\s*$", q)
    if m:
        lat, lon = float(m.group(1).replace(",", ".")), float(m.group(2).replace(",", "."))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            out.append(box_place(lat, lon, 0.5, f"{lat:.2f}, {lon:.2f} çevresi"))
    nq = norm(q)
    for key, meta in (BOLGE.items() if isinstance(BOLGE, dict) else []):
        if nq and (nq in norm(meta.get("ad")) or nq in norm(key)) and meta.get("kutu"):
            w, s_, e, n = map(float, meta["kutu"].split(","))
            out.append({"ad": meta.get("ad", key), "tam": meta.get("olay", "") or meta.get("ad", key), "tur": "kutu",
                        "ulke": "", "bbox": [w, s_, e, n], "lat": (s_ + n) / 2, "lon": (w + e) / 2,
                        "geojson": bbox_geom(w, s_, e, n)})
    seen = set()
    for cp in sorted(GEO_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True) if nq else []:
        try:
            data = json.loads(cp.read_text("utf-8"))
        except Exception:
            continue
        for it in (data if isinstance(data, list) else [data]):
            if not isinstance(it, dict) or "error" in it:
                continue
            nm = it.get("name") or (it.get("display_name") or "").split(",")[0]
            if nq in norm(nm) or nq in norm(it.get("display_name")):
                k = it.get("osm_id") or it.get("display_name")
                if k in seen:
                    continue
                seen.add(k)
                out.append(place_obj(it))
        if len(out) >= 8:
            break
    return out[:8]


def box_place(lat, lon, half, name):
    w, s, e, n = lon - half, lat - half, lon + half, lat + half
    return {"ad": name, "tam": f"{lat:.2f}° enlem, {lon:.2f}° boylam çevresinde {2 * half:g}° × {2 * half:g}° kutu "
                               f"(internet yok; idari sınır yerine kutu)", "tur": "kutu", "ulke": "",
            "bbox": [w, s, e, n], "lat": lat, "lon": lon, "geojson": bbox_geom(w, s, e, n)}


def place_obj(it):
    bb = [float(x) for x in it.get("boundingbox", [0, 0, 0, 0])]          # s, n, w, e
    w, s, e, n = bb[2], bb[0], bb[3], bb[1]
    geom = it.get("geojson") or {}
    if geom.get("type") not in ("Polygon", "MultiPolygon"):
        pad = 0.15                                                          # nokta/cizgi sonuc: kucuk kutu
        cx, cy = float(it.get("lon", (w + e) / 2)), float(it.get("lat", (s + n) / 2))
        w, s, e, n = min(w, cx - pad), min(s, cy - pad), max(e, cx + pad), max(n, cy + pad)
        geom = bbox_geom(w, s, e, n)
    addr = it.get("address") or {}
    name = it.get("name") or (it.get("display_name") or "").split(",")[0]
    return {"ad": name, "tam": it.get("display_name", name), "tur": it.get("addresstype") or it.get("type") or "",
            "ulke": addr.get("country", ""), "bbox": [w, s, e, n], "lat": float(it.get("lat", (s + n) / 2)),
            "lon": float(it.get("lon", (w + e) / 2)), "geojson": geom}


def search(q):
    try:
        res = nominatim("search", {"q": q, "format": "jsonv2", "polygon_geojson": 1, "polygon_threshold": 0.005,
                                   "addressdetails": 1, "limit": 8, "accept-language": "tr,en"})
    except NoNet:
        out = offline_search(q)
        if not out:
            raise RuntimeError("İnternet yok ve bu yer daha önce aranmamış. Haritadan tıkla, kutu çiz ya da "
                               "\"enlem, boylam\" yaz (ör. 38.4, 27.1).")
        return out
    out = [place_obj(it) for it in res]
    out.sort(key=lambda p: 0 if p["geojson"]["type"] != "Polygon" or len(p["geojson"]["coordinates"][0]) > 5 else 1)
    return out


def reverse(lat, lon, level):
    try:
        it = nominatim("reverse", {"lat": round(lat, 4), "lon": round(lon, 4), "zoom": LEVEL_ZOOM.get(level, 5),
                                   "format": "jsonv2", "polygon_geojson": 1, "polygon_threshold": 0.005,
                                   "addressdetails": 1, "accept-language": "tr,en"})
    except NoNet:
        half = {"ulke": 3.0, "il": 0.75, "ilce": 0.25, "sehir": 0.15}.get(level, 0.75)
        return box_place(lat, lon, half, f"Seçilen nokta çevresi ({LEVEL_NAME.get(level, level)} boyutunda)")
    if not it or "error" in it:
        raise RuntimeError("Bu noktada bir yerleşim sınırı bulunamadı (deniz olabilir).")
    return place_obj(it)


# ----------------------------------------------------------------------------
# 4. Harmonizasyon (build_app ile ayni yontem)
# ----------------------------------------------------------------------------
GROUPS = ["Terra", "Aqua", "VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21"]
VIIRS_ALL = ["VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21"]
VIIRS_AFTER = ["VIIRS_NOAA20", "VIIRS_NOAA21"]
SNPP_END = date(2026, 11, 2)            # FIRMS S-NPP VIIRS uretimi 2 Kasim 2026 13:00 UTC'de durdu (Earthdata duyurusu)
SNPP_END_D = (pd.Timestamp(SNPP_END) - pd.Timestamp("2000-01-01")).days   # ba.DAY0'a gore gun sirasi
DEFAULT = {}


def default_curve():
    """Yedek egri: Turkiye cok yillik verisinden (2012-2017) ogrenilen egri; yoksa 30 gunluk egri."""
    if DEFAULT:
        return DEFAULT["x"], DEFAULT["y"], DEFAULT["hi"], DEFAULT["k"]
    x = y = None
    hi, k = None, 1.3
    try:
        files = sorted((HERE / "dataset").glob("TR_[0-9][0-9][0-9][0-9].csv.gz"))
        if files:
            parts = []
            for f in files:
                yr = int(f.name[3:7])
                if not 2012 <= yr <= 2017:
                    continue
                d = pd.read_csv(f)
                act = [c for c in ("n_VIIRS_SNPP", "n_VIIRS_NOAA20", "n_VIIRS_NOAA21") if d[c].sum() > 0]
                parts.append(pd.DataFrame({"M": d["n_Terra"] + d["n_Aqua"], "V": d[act].sum(axis=1) / len(act)}))
            r = pd.concat(parts, ignore_index=True)
            fit = ba.fit_curve(r)
            if fit is not None:
                x, y = fit
                hi = 1.0
                k = float(r["M"].sum() / ba.f_curve(r["V"], x, y, hi).sum())
    except Exception as e:
        log("[bilgi] yedek eğri veri setinden kurulamadı:", e)
    if x is None:
        curve = pd.read_csv(HERE / "cikti_TR" / "transfer_curve.csv")
        x, y = ba.comb_curve(curve)
    DEFAULT.update(x=np.asarray(x, float), y=np.asarray(y, float), hi=hi, k=k)
    return DEFAULT["x"], DEFAULT["y"], DEFAULT["hi"], DEFAULT["k"]


def modis_only_static(modis, n_days, atlas):
    from sklearn.neighbors import BallTree
    flag = np.zeros(len(modis), bool)
    if not len(modis):
        return flag
    Xm = np.radians(modis[["lat", "lon"]].to_numpy(float))
    rad = 1.0 / h.EARTH_RADIUS_KM
    if atlas:
        A = np.unique(np.round(np.vstack(atlas), 3), axis=0)
        dist, _ = BallTree(np.radians(A), metric="haversine").query(Xm, k=1)
        flag |= dist[:, 0] <= rad
    t = pd.to_datetime(modis["timestamp"])
    dm = (t.dt.normalize() - t.min().normalize()).dt.days.to_numpy()
    idx = BallTree(Xm, metric="haversine").query_radius(Xm, r=rad)
    nd = np.array([len(np.unique(dm[ii])) for ii in idx])
    flag |= nd >= max(h.STATIC_MIN_DAYS, int(np.ceil(0.10 * n_days)))
    return flag


def aggregate(det):
    d = det
    gun = pd.to_datetime(d["timestamp"]).dt.normalize()
    ci = np.floor(d["lat"].to_numpy() / CELL).astype(int)
    cj = np.floor(d["lon"].to_numpy() / CELL).astype(int)
    t = pd.DataFrame({"d": (gun - ba.DAY0).dt.days.to_numpy(), "ci": ci, "cj": cj, "g": d["grup"].to_numpy()})
    n = t.groupby(["d", "ci", "cj", "g"]).size().unstack("g", fill_value=0)
    return n.reindex(columns=GROUPS, fill_value=0).reset_index()


def payload(rows, name, place, extra_note):
    rows = rows[(rows["M"] > 0) | (rows["V"] > 0)].copy()
    if rows.empty:
        raise RuntimeError("Bu bölgede ve dönemde hiç yangın tespiti yok.")
    rows["yil"] = (ba.DAY0 + pd.to_timedelta(rows["d"], unit="D")).dt.year
    x, y, hi, k_def = default_curve()
    how, curve = "Türkiye çok yıllı eğrisi (bölgeye özgü eğri için tespit az)", None
    ov = ba.overlap_years(rows)
    if len(ov) >= 3:
        a_, t_, dr_ = ba.split_years(ov)
        fit = ba.fit_curve(rows[rows["yil"].isin(a_)], min_n=12)
    else:
        a_, t_, dr_ = [], [], []
        sub = rows[rows["yil"].isin(ov)]
        fit = None
        if len(sub):
            days = np.sort(sub["d"].unique())
            fit = ba.fit_curve(sub[sub["d"] < days[int(len(days) * 0.6)]], min_n=12)
    if fit is not None:
        x, y = fit
        hi, how = 1.0, "bölgeye özgü eğri"
        curve = {"x": np.round(x, 4).tolist(), "y": np.round(y, 5).tolist(), "hi": 1.0}
    rows["f"] = ba.f_curve(rows["V"], x, y, hi)
    k, calib, ratio = ba.calibrate(rows, k_def)
    extra = ""
    if t_:
        te = rows["yil"].isin(t_)
        dd = rows[te].groupby("d").agg(M=("M", "sum"), f=("f", "sum"))
        if len(dd) > 2:
            extra = f"; test günlük seyir r={np.corrcoef(np.log1p(dd['M']), np.log1p(k * dd['f']))[0, 1]:.2f}"
        ww = rows[te].assign(w=rows["d"] // 7).groupby("w").agg(M=("M", "sum"), f=("f", "sum"))
        if len(ww) > 4:
            extra += f", haftalık r={np.corrcoef(np.log1p(ww['M']), np.log1p(k * ww['f']))[0, 1]:.2f}"
    if dr_:
        yy = rows[rows["yil"].isin(dr_)].groupby("yil").agg(M=("M", "sum"), f=("f", "sum"))
        yr = (yy["M"] / (k * yy["f"]).clip(lower=1e-9) * 100).round().astype(int)
        extra += "; yörünge kayması yıllarında ham MODIS ÷ uyumlu seri: " + ", ".join(f"{a} %{b}" for a, b in yr.items())
    if curve is None:
        curve = {"x": np.round(x, 4).tolist(), "y": np.round(y, 5).tolist(), "hi": hi}
    cells = rows[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    cix = {(a, b): i for i, (a, b) in enumerate(zip(cells["lat"], cells["lon"]))}
    unc = ba.year_unc(rows, k)
    n_tr = int(rows.loc[rows["yil"].isin(a_), "M"].sum()) if a_ else int(rows["M"].sum())
    guven = ("yüksek" if unc is not None and unc < 0.12 and n_tr >= 3000 else
             "orta" if unc is not None and unc < 0.25 and n_tr >= 800 else "düşük")
    calib_full = f"Kalibrasyon yıllarında {n_tr:,} MODIS tespiti. ".replace(",", ".") + f"{how}; {calib}{extra}"
    return {"unc": unc, "guven": guven,"key": "Q" + uuid.uuid4().hex[:8], "name": name, "cell": CELL, "k": round(float(k), 4),
            "calib": calib_full, "note": extra_note,
            "cities": [{"n": place.get("ad", name), "lat": place.get("lat"), "lon": place.get("lon")}] if place.get("lat") is not None else [],
            "cells": {"lat": cells["lat"].round(3).tolist(), "lon": cells["lon"].round(3).tolist()},
            "c": [cix[(a, b)] for a, b in zip(rows["lat"], rows["lon"])], "d": rows["d"].astype(int).tolist(),
            "m": rows["M"].astype(int).tolist(), "v": np.round(rows["V"] * 10).astype(int).tolist(),
            "curve": curve, "boundary": place.get("geojson")}


# ----------------------------------------------------------------------------
# 4b. Hizli yol: hazir veri seti (dataset/) kapsadigi bolgeler aninda hesaplanir
# ----------------------------------------------------------------------------
DATASETS = {}
DUNYA = {"years": set(), "dir": None}
TILE_CACHE = {}


def load_world():
    d = HERE / "dataset" / "DUNYA"
    DUNYA["years"], DUNYA["dir"] = set(), None
    TILE_CACHE.clear()
    m = d / "manifest.json"
    if m.exists():
        try:
            DUNYA["years"] = {int(y) for y in json.loads(m.read_text("utf-8")).get("yillar", {})}
            DUNYA["dir"] = d
            if DUNYA["years"]:
                log(f"dünya veri seti: {min(DUNYA['years'])}–{max(DUNYA['years'])} ({len(DUNYA['years'])} yıl)")
        except Exception as e:
            log("[bilgi] dünya manifesti okunamadı:", e)


def world_covers(y0, y1):
    return DUNYA["dir"] is not None and all(y in DUNYA["years"] for y in range(y0, y1 + 1))


def world_rows(bbox, y0, y1):
    w, s, e, n = bbox
    parts = []
    for yr in range(y0, y1 + 1):
        for ti in range(int(np.floor(s / 10)), int(np.floor(min(n, 89.99) / 10)) + 1):
            for tj in range(int(np.floor(w / 10)), int(np.floor(min(e, 179.99) / 10)) + 1):
                key = (ti, tj, yr)
                if key not in TILE_CACHE:
                    f = DUNYA["dir"] / f"T{ti}_{tj}_{yr}.csv.gz"
                    if f.exists():
                        dd = pd.read_csv(f)
                        TILE_CACHE[key] = pd.DataFrame({"lat": dd["lat_c"].round(3), "lon": dd["lon_c"].round(3),
                                                        "d": (pd.to_datetime(dd["gun"]) - ba.DAY0).dt.days,
                                                        "M": dd["n_Terra"] + dd["n_Aqua"],
                                                        # n_VIIRS_ort: dunya_guncelle.py'nin ekledigi gunler (S-NPP sonrasi NOAA ortalamasi)
                                                        "V": (dd["n_VIIRS_ort"].fillna(dd["n_VIIRS_SNPP"]) if "n_VIIRS_ort" in dd
                                                              else dd["n_VIIRS_SNPP"]).astype(float)})
                    else:
                        TILE_CACHE[key] = None
                    if len(TILE_CACHE) > 4000:
                        TILE_CACHE.pop(next(iter(TILE_CACHE)))
                if TILE_CACHE[key] is not None:
                    parts.append(TILE_CACHE[key])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["lat", "lon", "d", "M", "V"])


def load_datasets():
    vcols = ["n_VIIRS_SNPP", "n_VIIRS_NOAA20", "n_VIIRS_NOAA21"]
    DATASETS.clear()
    for key, bb in h.REGIONS.items():
        files = sorted((HERE / "dataset").glob(f"{key}_[0-9][0-9][0-9][0-9].csv.gz"))
        if not files:
            continue
        parts, years = [], set()
        for f in files:
            years.add(int(f.name[-11:-7]))
            d = pd.read_csv(f)
            if d.empty:
                continue
            act = [c for c in vcols if c in d.columns and d[c].sum() > 0]
            parts.append(pd.DataFrame({"lat": d["lat_c"].round(3), "lon": d["lon_c"].round(3),
                                       "d": (pd.to_datetime(d["gun"]) - ba.DAY0).dt.days,
                                       "M": d["n_Terra"] + d["n_Aqua"],
                                       "V": d[act].sum(axis=1) / len(act) if act else 0.0}))
        if parts:
            w, s, e, n = map(float, bb.split(","))
            meta = BOLGE.get(key, {})
            DATASETS[key] = {"bbox": (w, s, e, n), "years": years, "rows": pd.concat(parts, ignore_index=True),
                             "ad": meta.get("ad", key), "olay": meta.get("olay", "")}
            log(f"hazır veri seti {key}: {min(years)}–{max(years)}, {len(DATASETS[key]['rows']):,} hücre-gün")


def cell_fraction(lat_c, lon_c, rings, k=6):
    """Her 0.25 derece hucrenin sinir icinde kalan alan payi (k x k ornek nokta)."""
    off = (np.arange(k) + 0.5) / k * CELL - CELL / 2
    gx, gy = np.meshgrid(off, off)
    px = (lon_c[:, None] + gx.ravel()[None, :]).ravel()
    py = (lat_c[:, None] + gy.ravel()[None, :]).ravel()
    return points_in(px, py, rings).reshape(len(lat_c), -1).mean(axis=1)


def fast_region(rings, raw_bbox, y0, y1, place):
    w, s, e, n = raw_bbox
    by_area = sorted(DATASETS.items(), key=lambda kv: (kv[1]["bbox"][2] - kv[1]["bbox"][0]) * (kv[1]["bbox"][3] - kv[1]["bbox"][1]))
    if world_covers(y0, y1):
        by_area.append(("DUNYA", {"bbox": (-180.0, -90.0, 180.0, 90.0), "years": DUNYA["years"], "rows": None,
                                  "ad": "Dünya", "olay": ""}))
    for key, ds in by_area:
        W, S, E, N = ds["bbox"]
        if not (W - 1e-6 <= w and S - 1e-6 <= s and E + 1e-6 >= e and N + 1e-6 >= n):
            continue
        if not all(y in ds["years"] for y in range(y0, y1 + 1)):
            continue
        r = ds["rows"] if ds["rows"] is not None else world_rows(raw_bbox, y0, y1)
        if ds["rows"] is not None and len(r) and world_covers(y1, y1):
            # bolge veri seti eski bir gunde bitiyorsa (or. S-NPP kapanisi), sonrasini guncellenmis dunya verisinden ekle
            last = int(r["d"].max())
            wr = world_rows(raw_bbox, y1, y1)
            wr = wr[wr["d"] > last]
            if len(wr):
                r = pd.concat([r, wr[r.columns.intersection(wr.columns)]], ignore_index=True)
        if r.empty:
            raise RuntimeError("Bu bölgede ve dönemde hiç yangın tespiti yok.")
        d0, d1 = (pd.Timestamp(y0, 1, 1) - ba.DAY0).days, (pd.Timestamp(y1, 12, 31) - ba.DAY0).days
        r = r[(r["d"] >= d0) & (r["d"] <= d1) & (r["lon"] >= w - CELL) & (r["lon"] <= e + CELL) &
              (r["lat"] >= s - CELL) & (r["lat"] <= n + CELL)]
        cells = r[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
        if cells.empty:
            raise RuntimeError("Bu bölgede ve dönemde hiç yangın tespiti yok.")
        cells["f"] = cell_fraction(cells["lat"].to_numpy(float), cells["lon"].to_numpy(float), rings)
        keep = cells[cells["f"] >= 0.5]
        rule = "alanının en az yarısı sınırın içinde kalan 0,25° hücreler"
        if keep.empty:
            keep = cells[cells["f"] == cells["f"].max()]
            keep = keep[keep["f"] > 0]
            rule = "bölge bir hücreden küçük; en çok örtüşen 0,25° hücre"
        if keep.empty:
            raise RuntimeError("Bu bölgede ve dönemde hiç yangın tespiti yok.")
        rows = r.merge(keep[["lat", "lon"]], on=["lat", "lon"])
        yrs = sorted(ds["years"])
        note = (f"Hazır veri setinden anında hesaplandı ({ds['ad']} veri seti {yrs[0]}–{yrs[-1]}, MODIS + VIIRS, "
                f"kalıcı kaynaklar ayıklanmış). Kullanılan: {rule}, {len(keep)} hücre.")
        return payload(rows, place.get("ad") or "Seçilen alan", place, note), key
    return None, None


try:
    from bolgeler import HEPSI as BOLGE
except ImportError:
    BOLGE = {}


# ----------------------------------------------------------------------------
# 5. Isler (arka planda calisan bolge hesaplari)
# ----------------------------------------------------------------------------
JOBS = {}
AVAIL = {}


class Job:
    def __init__(self, spec):
        self.id = uuid.uuid4().hex[:12]
        self.spec = spec
        self.state, self.msg = "kuyrukta", "sıraya alındı"
        self.done = self.total = self.cached = 0
        self.result = None
        self.t0 = time.time()
        self.lock = threading.Lock()

    def tick(self, cached):
        with self.lock:
            self.done += 1
            self.cached += int(cached)

    def status(self):
        return {"durum": self.state, "mesaj": self.msg, "yapilan": self.done, "toplam": self.total,
                "onbellek": self.cached, "sure": round(time.time() - self.t0)}


def run_job(job):
    try:
        _run_job(job)
    except Exception as e:
        traceback.print_exc()
        job.state, job.msg = "hata", str(e)


def _run_job(job):
    sp = job.spec
    place = sp.get("yer") or {}
    geom = place.get("geojson") or (bbox_geom(*sp["bbox"]) if sp.get("bbox") else None)
    if not geom:
        raise RuntimeError("Bölge sınırı yok.")
    place["geojson"] = geom
    rings = rings_of(geom)
    if not rings:
        raise RuntimeError("Sınır geometrisi okunamadı.")
    raw_bbox = geom_bbox(rings)
    y0_, y1_ = int(sp.get("y0", 2012)), int(sp.get("y1", 2100))
    if (raw_bbox[2] - raw_bbox[0] > MAX_SPAN_DEG or raw_bbox[3] - raw_bbox[1] > MAX_SPAN_DEG) and \
            not world_covers(max(2003, min(y0_, y1_)), min(datetime.now(timezone.utc).year, max(y0_, y1_))):
        raise RuntimeError(f"Bölge çok geniş ({raw_bbox[2] - raw_bbox[0]:.0f}° × {raw_bbox[3] - raw_bbox[1]:.0f}°). "
                           f"{MAX_SPAN_DEG}°'den küçük bir alan seç (180. meridyeni kesen ülkeler de bu yüzden reddedilir).")
    bbox = snap_bbox(raw_bbox)
    today = datetime.now(timezone.utc).date()
    y0, y1 = int(sp.get("y0", 2012)), int(sp.get("y1", today.year))
    y0, y1 = max(2003, min(y0, y1)), min(today.year, max(y0, y1))
    tum_v = bool(sp.get("tum_viirs"))
    sensors_v = list(VIIRS_ALL) if tum_v else ["VIIRS_SNPP"]
    if not tum_v and min(date(y1, 12, 31), today - timedelta(days=1)) >= SNPP_END:
        sensors_v += VIIRS_AFTER          # S-NPP kapandi: 2 Kasim 2026 sonrasi NOAA-20/21 ile devam

    def year_sensors(yr):
        """O yil icin indirilecek VIIRS uydulari (S-NPP kapanisindan sonraki yillar icin NOAA-20/21 eklenir)."""
        if tum_v:
            return list(VIIRS_ALL)
        end = min(date(yr, 12, 31), today - timedelta(days=1))
        return ["VIIRS_SNPP"] + (VIIRS_AFTER if end >= SNPP_END else [])
    if not sp.get("canli"):
        job.msg = "hazır veri setinde aranıyor"
        obj, dkey = fast_region(rings, raw_bbox, y0, y1, place)
        if obj is not None:
            obj["name"] = place.get("ad") or "Seçilen alan"
            obj["donem"] = f"{y0}–{y1}" if y0 != y1 else str(y0)
            job.result = gzip.compress(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 6)
            job.total = job.done = 1
            job.state, job.msg = "bitti", f"{len(obj['d']):,} hücre-gün hazır veri setinden ({time.time() - job.t0:.1f} sn)".replace(",", ".")
            log(f"[iş {job.id}] {place.get('ad', '?')} {y0}-{y1}: hazır veri seti {dkey}, {job.msg}")
            return
    key = h.get_map_key()
    if not AVAIL:
        AVAIL.update(h.get_availability(key))

    # plan: her yil, her sensor, 5 gunluk pencereler (build_dataset ile ayni hizalama -> ortak onbellek)
    plan = []
    for yr in range(y0, y1 + 1):
        start, end = date(yr, 1, 1), min(date(yr, 12, 31), today - timedelta(days=1))
        for sensor in ["MODIS"] + year_sensors(yr):
            cur = start
            while cur <= end:
                nd = min(5, (end - cur).days + 1)
                for sid, st, n in h.plan_window(sensor, cur, nd, AVAIL):
                    plan.append((yr, sensor, sid, st, n))
                cur += timedelta(days=nd)
    job.total = len(plan)
    job.state, job.msg = "calisiyor", f"FIRMS'ten {len(plan)} pencere alınıyor (kutu {bbox_str(bbox)})"
    log(f"[iş {job.id}] {place.get('ad', '?')}  {y0}-{y1}  kutu {bbox_str(bbox)}  {len(plan)} pencere")

    frames = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_window, key, sid, bbox, st, n): (yr, sensor) for yr, sensor, sid, st, n in plan}
        for fu in as_completed(futs):
            df, cached = fu.result()
            job.tick(cached)
            if len(df):
                frames.setdefault(futs[fu], []).append(df)
            if job.done % 25 == 0:
                job.msg = f"FIRMS: {job.done}/{job.total} pencere ({job.cached} önbellekten)"

    # yillari yeniden eskiye isle: VIIRS yillarinin kalici kaynak atlasi MODIS-tek yillara tasinir
    atlas, tables, n_static, n_pts = [], [], 0, 0
    for yr in range(y1, y0 - 1, -1):
        job.msg = f"{yr} işleniyor"
        mraw = pd.concat(frames.get((yr, "MODIS"), []), ignore_index=True) if frames.get((yr, "MODIS")) else pd.DataFrame()
        modis = h.standardize(mraw, "MODIS") if len(mraw) else pd.DataFrame(columns=["lat", "lon", "timestamp", "satellite", "source"])
        vparts = []
        for s in year_sensors(yr):
            if frames.get((yr, s)):
                vparts.append(h.standardize(pd.concat(frames[(yr, s)], ignore_index=True), s))
        viirs = pd.concat(vparts, ignore_index=True) if vparts else pd.DataFrame(columns=modis.columns)
        if len(modis):
            modis = modis[points_in(modis["lon"].to_numpy(float), modis["lat"].to_numpy(float), rings)].reset_index(drop=True)
            modis["satellite"] = modis["satellite"].astype(str).replace({"T": "Terra", "A": "Aqua"})
        if len(viirs):
            viirs = viirs[points_in(viirs["lon"].to_numpy(float), viirs["lat"].to_numpy(float), rings)].reset_index(drop=True)
        if modis.empty and viirs.empty:
            continue
        if yr == today.year and len(modis) and len(viirs):
            modis, viirs = h.trim_incomplete(modis, viirs)
        sm, sv = np.zeros(len(modis), bool), np.zeros(len(viirs), bool)
        if len(modis) and len(viirs):
            sm, sv, *_ = h.flag_static(modis, viirs)
            if sv.any():
                atlas.append(viirs.loc[sv, ["lat", "lon"]].to_numpy(float))
        elif len(modis):
            end = min(date(yr, 12, 31), today - timedelta(days=1))
            sm = modis_only_static(modis, (end - date(yr, 1, 1)).days + 1, atlas)
        n_static += int(sm.sum() + sv.sum())
        mc, vc = modis[~sm], viirs[~sv]
        n_pts += len(mc) + len(vc)
        det = pd.concat([mc.assign(grup=mc["satellite"].astype(str)), vc.assign(grup=vc["source"].astype(str))],
                        ignore_index=True)
        if det.empty:
            continue
        t = aggregate(det)
        t["M"] = t["Terra"] + t["Aqua"]
        if tum_v:
            active = [g for g in sensors_v if t[g].sum() > 0]
            t["V"] = t[active].sum(axis=1) / len(active) if active else 0.0
        else:
            # Kalibrasyon S-NPP ile yapildi; S-NPP'nin kapandigi gunden itibaren (2 Kasim 2026) V = aktif NOAA uydularinin
            # ortalamasi. Aktiflik gun gun belirlenir: o gun bolgede tespiti olan NOAA uydulari.
            after = t["d"].to_numpy() >= SNPP_END_D
            v = t["VIIRS_SNPP"].to_numpy(float).copy()
            if after.any():
                noaa = t[VIIRS_AFTER]
                has = noaa.groupby(t["d"]).transform("sum") > 0          # o gun bolgede tespit veren uydu
                n_act = has.sum(axis=1).to_numpy()
                mean_noaa = np.where(n_act > 0, (noaa * has).sum(axis=1).to_numpy() / np.maximum(n_act, 1), 0.0)
                v = np.where(after, mean_noaa, v)
            t["V"] = v
        tables.append(t[["d", "ci", "cj", "M", "V"]])
    if not tables:
        raise RuntimeError("Bu bölgede ve dönemde hiç yangın tespiti yok.")
    tab = pd.concat(tables, ignore_index=True)
    rows = pd.DataFrame({"lat": (tab["ci"] + 0.5) * CELL, "lon": (tab["cj"] + 0.5) * CELL, "d": tab["d"],
                         "M": tab["M"], "V": tab["V"]})
    tr = lambda v: f"{v:,}".replace(",", ".")
    note = (f"Canlı hesap: {tr(job.total)} FIRMS penceresi ({tr(job.cached)} önbellekten), sınır içinde {tr(n_pts)} tespit, "
            f"{tr(n_static)} kalıcı kaynak tespiti ayıklandı. VIIRS uyduları: " + ", ".join(s.replace("VIIRS_", "") for s in sensors_v) + "."
            + ("" if tum_v or sensors_v == ["VIIRS_SNPP"] else
               " S-NPP 2 Kasım 2026'da kapandığı için o günden sonrası NOAA-20/21 ortalamasıyla hesaplandı."))
    obj = payload(rows, place.get("ad") or "Seçilen alan", place, note)
    obj["name"] = place.get("ad") or "Seçilen alan"
    obj["donem"] = f"{y0}–{y1}" if y0 != y1 else str(y0)
    job.result = gzip.compress(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 6)
    job.state, job.msg = "bitti", f"{len(rows):,} hücre-gün hazır ({round(time.time() - job.t0)} sn)"
    log(f"[iş {job.id}] bitti: {len(rows):,} hücre-gün, {len(job.result) / 1e6:.1f} MB")


# ----------------------------------------------------------------------------
# 5b. Harita karolari: sunucu uzerinden, diskte onbellekli (API anahtari gerekmez; demo internetsiz de calisir)
# ----------------------------------------------------------------------------
TILE_DIR = h.CACHE_DIR / "karolar"
TILE_SOURCES = [
    "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
]
TILE_LOCK = threading.Semaphore(2)          # OSM kullanim kurali: az sayida eszamanli istek


def get_tile(z, x, y):
    if not (0 <= z <= 13 and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
        raise ValueError("karo dışı")
    cp = TILE_DIR / str(z) / str(x) / f"{y}.png"
    if cp.exists():
        return cp.read_bytes()
    if OFFLINE:
        return None
    last = None
    with TILE_LOCK:
        for src in TILE_SOURCES:
            try:
                r = requests.get(src.format(z=z, x=x, y=y), headers=UA, timeout=20)
                if r.status_code == 200 and r.content[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1") or \
                        (r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image/")):
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    cp.write_bytes(r.content)
                    return r.content
                last = f"{r.status_code}"
            except Exception as e:
                last = str(e)
    log(f"[karo] {z}/{x}/{y} alınamadı: {last}")
    return None


# ----------------------------------------------------------------------------
# 6. HTTP
# ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "AtesTakvimi/1.0"

    def log_message(self, fmt, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", gz=False):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._send(code, {"hata": msg})

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                p = HERE / "app" / "ates_takvimi.html"
                if not p.exists():
                    return self._send(404, "Önce: python build_app.py", "text/plain; charset=utf-8")
                return self._send(200, p.read_bytes(), "text/html; charset=utf-8")
            if u.path.startswith("/yerel/"):
                # internetsiz demo icin yerel kopyalar (Leaflet, yazi tipleri): python cevrimdisi_hazirla.py
                base = (HERE / "app" / "yerel").resolve()
                f = (base / u.path[len("/yerel/"):]).resolve()
                if base not in f.parents or not f.is_file():
                    return self._send(404, b"", "text/plain")
                ct = {".js": "application/javascript", ".css": "text/css; charset=utf-8", ".png": "image/png",
                      ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf"}.get(f.suffix.lower(),
                                                                                          "application/octet-stream")
                return self._send(200, f.read_bytes(), ct)
            if u.path.startswith("/karo/"):
                z, x, y = [int(v) for v in u.path[len("/karo/"):].removesuffix(".png").split("/")]
                data = get_tile(z, x, y)
                if data is None:
                    return self._send(404, b"", "image/png")
                self.send_response(200)
                self.send_header("Content-Type", "image/png" if data[:4] == b"\x89PNG" else "image/jpeg")
                self.send_header("Cache-Control", "max-age=604800")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if u.path == "/api/veri_setleri":
                if q.get("yenile"):
                    load_datasets()
                    load_world()
                out = [{"key": k, "ad": d["ad"], "olay": d["olay"], "kutu": list(d["bbox"]),
                        "y0": min(d["years"]), "y1": max(d["years"])} for k, d in DATASETS.items()]
                if DUNYA["years"]:
                    out.append({"key": "DUNYA", "ad": "Dünya (tüm ülkeler)", "olay": "", "kutu": [-180, -90, 180, 90],
                                "y0": min(DUNYA["years"]), "y1": max(DUNYA["years"])})
                return self._send(200, out)
            if u.path == "/api/durum":
                return self._send(200, {"ok": True, "cevrimdisi": OFFLINE, "anahtar": (HERE / "map_key.txt").exists(),
                                        "son_yil": datetime.now(timezone.utc).year, "paralel": WORKERS})
            if u.path == "/api/ara":
                if not q.get("q", "").strip():
                    return self._err(400, "arama metni boş")
                return self._send(200, search(q["q"].strip()[:200]))
            if u.path == "/api/ters":
                return self._send(200, reverse(float(q["lat"]), float(q["lon"]), q.get("duzey", "il")))
            if u.path == "/api/is":
                job = JOBS.get(q.get("id", ""))
                return self._send(200, job.status()) if job else self._err(404, "iş bulunamadı")
            if u.path == "/api/sonuc":
                job = JOBS.get(q.get("id", ""))
                if not job or job.result is None:
                    return self._err(404, "sonuç hazır değil")
                return self._send(200, job.result, gz=True)
            return self._err(404, "yok")
        except requests.HTTPError as e:
            return self._err(502, f"dış servis hatası: {e}")
        except Exception as e:
            traceback.print_exc()
            return self._err(500, str(e))

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            spec = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            if u.path == "/api/bolge":
                job = Job(spec)
                JOBS[job.id] = job
                threading.Thread(target=run_job, args=(job,), daemon=True).start()
                return self._send(200, {"id": job.id})
            return self._err(404, "yok")
        except Exception as e:
            traceback.print_exc()
            return self._err(500, str(e))


def main():
    global CACHE
    html, src = HERE / "app" / "ates_takvimi.html", HERE / "build_app.py"
    if not html.exists() or src.stat().st_mtime > html.stat().st_mtime:
        print("Uygulama sayfası eski ya da yok; yeniden üretiliyor (python build_app.py)...", flush=True)
        import subprocess
        subprocess.run([sys.executable, str(src)], cwd=HERE)
    if not html.exists():
        sys.exit("app/ates_takvimi.html üretilemedi. Elle dene: python build_app.py")
    if not OFFLINE and not (HERE / "map_key.txt").exists():
        print("Uyarı: map_key.txt yok. Hazır bölgeler ve dünya verisi çalışır; FIRMS'ten canlı indirme yapılamaz.")
    CACHE = CacheIndex()
    default_curve()
    load_datasets()
    load_world()
    n_files = sum(len(v) for v in CACHE.idx.values())
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Ateş Takvimi sunucusu: {url}   (önbellekte {n_files} FIRMS penceresi; durdurmak için Ctrl+C)")
    if "--tarayici-yok" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nkapatıldı")


if __name__ == "__main__":
    main()
