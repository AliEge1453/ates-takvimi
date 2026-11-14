"""
MODIS / VIIRS Hot Spot Harmonization - tek dosya (v0.13)
=======================================================
Calistir:   python harmonize_v13.py                          (Turkiye, son 5 gun)
            python harmonize_v13.py --days 30                 (son 30 gun)
            python harmonize_v13.py --region LA --start 2025-01-01 --end 2025-01-31   (LA yanginlari, arsiv)
            python harmonize_v13.py --region CA --days 30     (Kaliforniya, son 30 gun)
            python harmonize_v13.py --bbox -120,34,-118,35 --start ... --end ...      (ozel kutu)
            python harmonize_v13.py --refresh                 (onbellegi yoksay)
Bolgeler: TR, CA, LA, NSW, PANTANAL, GREECE, PORTUGAL (asagida REGIONS; istedigini ekleyebilirsin)

(map_key.txt ayni klasorde varsa onu kullanir, yoksa key'i sorar.)

v0.13 yenilikleri:
  - UYUM MERDIVENI: ayni veriyle 5 olcekte MODIS<->VIIRS uyumu (piksel-gecis, olay, hucre-gun 0.25/0.5
    derece, +-1 gun toleransi, hucre-hafta ve mekansal/zamansal korelasyon). Harmonizasyonun HANGI olcekte
    gecerli oldugunu gosterir. Sonuc: uyum_merdiveni.csv
  - YANGIN KALICILIGI: VIIRS yangin olaylari kac ayri gecis boyunca goruldu -> MODIS'in yakalama orani.
    Kisa omurlu (anız) yanginlarin neden piksel duzeyinde eslesmedigini olcer. Sonuc: kalicilik.csv
  - Referanslar: Fu ve ark. 2020 (Remote Sensing 12:2870), Sofiev & Kouznetsov 2026 (GMD 19:10123)
  (v0.12'den:)
  - BOLGELER: --region TR/CA/LA/NSW/PANTANAL/GREECE/PORTUGAL ya da --bbox bati,guney,dogu,kuzey.
    Cikti klasoru bolgeye gore (cikti_<BOLGE>).
  - TARIH ARALIGI: --start ve --end (YYYY-MM-DD). Gecmis tarihler icin FIRMS arsivi (SP), yakin tarihler
    icin NRT otomatik secilir (FIRMS data_availability tablosundan); sinirda ikisi birlestirilir.
    Henuz calismayan uydular (ornegin 2018 oncesi NOAA-20) atlanir.
  - SP verisi onbellekte kalici (tarih degismez), NRT gunluk.
  (v0.11'den:)
  - EKSIK SON GUNLERI ATAR: bir kaynagin (orn. SNPP) verisi digerlerinden geride ise ya da bugun yarimsa,
    o gunler analizden cikarilir (onceki surumde son gunler VIIRS eksik oldugu icin testi bozuyordu).
  - Kalici kaynak ESIK DUYARLILIGI tablosu (2..25 gun): esik secimi sayilarla gorunur.
  - Donusum modeline 3. girdi: o gunun toplam VIIRS aktivitesi ("rejim"); D2_model_gunluk.
  (v0.10'dan:)
  - GUNLUK VERI KALITESI: gun x uydu tespit sayilari, uydu bazinda eslesme, ani dusen uydu (veri
    boslugu suphesi) bayraklari. Bayrakli gunlerde ve digerlerinde eslesme ayri yazilir.
  - VIIRS -> MODIS DONUSUM MODELI: hucre-gunde VIIRS sayisi/FRP'sinden MODIS (Aqua/Terra) sayisi tahmini.
    Zaman ayrimli dogrulama (ilk 2/3 gun egitim, son 1/3 test), sabit-carpan ile karsilastirma,
    kalibrasyon tablosu ve transfer_curve.csv (VIIRS aktivitesi -> beklenen MODIS sayisi / P(>=1)).
  (v0.9'dan:)
  - UZUN VERI: --days N ile son N gun (FIRMS tek istekte en fazla 5 gun verdigi icin 5'er gunluk
    pencerelerle, DATE parametresiyle). Her pencere ayri onbelleklenir. Uzun pencerede kalici kaynak
    kurali oransal (>= %50 gun), tablolar haftalik ozetlenir.
  - SEYRELTILMIS AYNI-SENSOR KONTROLU: VIIRS, MODIS'in seyrek yogunluguna indirilip diger VIIRS ile
    korele edilir. MODIS~VIIRS korelasyonunun ne kadari sadece "az nokta" gurultusu, ne kadari
    gercek zaman/sensor farki? (gercek/kontrol orani)
  - Korelasyon tablosuna "Zamansal_log1p": hucre ortalamasi cikarilinca (konum etkisi yok) kalan uyum
  - Asama sureleri ekrana yazilir
  (v0.8'den:)
  - SAGLAM INDIRME: baglanti kopmasinda 3 kez tekrar dener; indirilen veriyi data_cache/ klasorune
    kaydeder (ayni gun tekrar calistirinca ayni veri kullanilir = tekrarlanabilirlik, internet gerekmez).
    Bir VIIRS uydusu alinamazsa sessizce eksik analiz yapmaz, durur (ALLOW_PARTIAL ile degistirilebilir).
  - IZGARA (hucre-gun) KARSILASTIRMASI: nokta eslestirme yerine 0.25 derece x gun toplamlari:
    sayi/FRP korelasyonu, "VIIRS aktivitesine gore MODIS'in gorme olasiligi", gunluk oran (kalibrasyon).
  - Ekrandan kaldirilanlar: lojistik esik tahmini, guven/gunduz-gece kirilimi (bilgi vermiyordu)
  (v0.7'den:)
  - KALICI KAYNAK filtresi: ayni yerde >= 4 farkli gunde tekrar eden tespitler
    (sanayi flare'i, kalici termal kaynak adaylari) isaretlenir, istatistikten cikarilir
  - TESPIT VERIMLILIGI egrisi: VIIRS olayinin FRP'si buyudukce MODIS (Aqua/Terra) onu
    gorme olasiligi nasil artiyor; %25/%50/%75 olasilik esikleri (lojistik uyum)
  - FRP KALIBRASYONU: eslesen MODIS FRP / ayni VIIRS gecisinin cakisan piksel FRP toplami
  - Uyum matrisine "VIIRS(diger)" sutunu (kendisi haric tum VIIRS uydulari birlesik)
  (v0.6'dan:)
  - GECIS SAATI tablosu: her uydunun gercek yerel gunes saati (Terra/Aqua yorunge kaymasi!)
  - TUM-CIFTLER uyum matrisi: Terra, Aqua, VIIRS-SNPP/N20/N21 arasinda hangisi hangisini
    ne kadar goruyor + medyan zaman farki. VIIRS<->VIIRS = "ayni sensor, farkli zaman" kontrolu.
  - Gunluk rapora VIIRS nokta sayisi, bolge raporuna zaman farki ve Aqua payi eklendi
  - Eski sabit-esik duyarlilik tablosu ekrandan kaldirildi (sensitivity.csv hala yazilir)
  (v0.5'ten:)
  - Terra / Aqua icin 1, 3, 6, 12 saatlik pencerelerde AYRI eslesme tablosu
    (Aqua <=1 saat sonucu "saf sensor/algoritma farki"na en yakin olcumdur)
  - Daha uzun veri penceresi (10 -> 5 -> 3 gun, hangisi kabul edilirse) ve gunluk kirilim
  - MODIS noktalari da olay kumelerine ayrilir; olay duzeyinde eslesme + %95 guven araligi
  (v0.4'ten:)
  - Piksel ayak izine dayali eslestirme: sabit 1.5 km yerine, her ciftin
    gercek piksel boyutlari (scan/track) kullanilir. Esleme kosulu:
    merkez mesafesi <= MODIS_yarim_boyut + VIIRS_yarim_boyut
    (piksel yonelimi bilinmedigi icin kare/daire yaklasimi).
  - VIIRS tekilleştirme: 3 VIIRS uydusunun ayni yangini tekrar tekrar
    saymasi onlenir (0.75 km / 3 saat icindeki tespitler tek "olay" kumesi).
  - Terra / Aqua ayri rapor (eslesme, zaman farki) + UTC saatine gore kirilim
  - Bolge (1 derece hucre) bazli eslesme tablosu
"""

import sys
import time
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

# ----------------------------------------------------------------------------
# AYARLAR
# ----------------------------------------------------------------------------
def _argval(flag, default):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


# bati,guney,dogu,kuzey (derece). Istedigin bolgeyi buraya ekleyebilirsin.
REGIONS = {
    "TR": "25,35,45,43",                  # Turkiye ve cevresi
    "CA": "-124.6,32.4,-114.0,42.1",      # Kaliforniya
    "LA": "-119.0,33.6,-117.4,34.6",      # Los Angeles havzasi (Ocak 2025: Palisades, Eaton yanginlari)
    "NSW": "140.9,-37.6,153.7,-28.0",     # Yeni Guney Galler, Avustralya (2019-20 "Black Summer")
    "PANTANAL": "-59.0,-22.0,-54.5,-15.0",  # Pantanal, Brezilya (2020)
    "GREECE": "19.3,34.8,28.4,41.8",      # Yunanistan
    "PORTUGAL": "-9.6,36.9,-6.1,42.2",    # Portekiz
}
REGION = _argval("--region", "TR").upper()
if _argval("--bbox", None):
    REGION = "OZEL"
    AREA_BBOX = _argval("--bbox", None).replace(" ", "")
elif REGION in REGIONS:
    AREA_BBOX = REGIONS[REGION]
else:
    sys.exit(f"Bilinmeyen bolge '{REGION}'. Secenekler: {', '.join(REGIONS)} (ya da --bbox bati,guney,dogu,kuzey)")
DATE = None                    # BITIS tarihi "YYYY-MM-DD" ya da None (bugun, UTC); komut satirinda --end
MAX_DISTANCE_KM = 1.5          # "ayni yangin" mesafe esigi
MAX_TIME_DELTA_HOURS = 6       # "ayni yangin" zaman esigi
EARTH_RADIUS_KM = 6371.0088

# Duyarlilik tablosu icin denenecek esikler
SENS_DISTANCES = [0.5, 1.0, 1.5, 2.0, 3.0]
SENS_HOURS = [3, 6, 12, 24]
WINDOWS = [1, 3, 6, 12]       # Terra/Aqua tablosu icin zaman pencereleri (saat)

STATIC_RADIUS_KM = 0.5         # "ayni yer" yaricapi
STATIC_MIN_DAYS = 4            # kac farkli gunde tekrar ederse kalici kaynak adayi (alt sinir)
STATIC_FRACTION = 0.5          # uzun pencerede: gunlerin en az bu orani kadar gunde tekrar
QC_MIN_VIIRS = 100             # gunluk VIIRS ort. tespit bundan azsa bayraklama yapma (dusuk aktivite)
QC_RATIO = 0.2                 # bir MODIS uydusunun VIIRS'e orani kendi medyaninin bu katindan azsa bayrakla
EXCLUDE_STATIC = True          # True: kalici kaynak adaylarini analizden cikar

GRID_DEG = 0.25               # izgara hucre boyutu (derece)
FETCH_RETRIES = 3             # baglanti hatasinda deneme sayisi
ALLOW_PARTIAL = False         # True: bir VIIRS uydusu alinamasa da devam et (onerilmez)
REFRESH = "--refresh" in sys.argv   # onbellegi yoksay, taze veri indir


def _argval(flag, default):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


TOTAL_DAYS = int(_argval("--days", 5))   # kac gunluk veri (bitis tarihinden geriye)
_T0 = time.time()


def _tick(msg: str):
    print(f"[{time.time() - _T0:5.0f} sn] {msg}", flush=True)

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "data_cache"
OUT_DIR = HERE / f"cikti_{REGION}"      # ciktilar bolgeye gore ayri klasorde


def get_map_key() -> str:
    key_file = HERE / "map_key.txt"
    if key_file.exists():
        key = key_file.read_text().strip()
        if key:
            return key
    key = input("FIRMS MAP_KEY'ini yapistir ve Enter'a bas: ").strip()
    if not key:
        sys.exit("Key girilmedi.")
    key_file.write_text(key)
    print(f"Key kaydedildi: {key_file}")
    return key


# ----------------------------------------------------------------------------
# 1. FETCH
# ----------------------------------------------------------------------------
def _end_date():
    e = _argval("--end", None) or DATE
    if e:
        return datetime.strptime(e, "%Y-%m-%d").date()
    return datetime.now(timezone.utc).date()


def _start_date():
    st = _argval("--start", None)
    if st:
        return datetime.strptime(st, "%Y-%m-%d").date()
    return _end_date() - timedelta(days=TOTAL_DAYS - 1)


REQUEST_PAUSE = float(_argval("--pause", 0.12))      # FIRMS limiti: 10 dakikada 5000 istek

# mantiksal sensor -> (arsiv SP kaynagi, yakin tarih NRT kaynagi)
SENSOR_IDS = {
    "MODIS": ("MODIS_SP", "MODIS_NRT"),
    "VIIRS_SNPP": ("VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"),
    "VIIRS_NOAA20": ("VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT"),
    "VIIRS_NOAA21": ("VIIRS_NOAA21_SP", "VIIRS_NOAA21_NRT"),
}
# FIRMS data_availability alinamazsa kullanilan son bilinen degerler
_today = datetime.now(timezone.utc).date()
DEFAULT_AVAIL = {
    "MODIS_SP": (date(2000, 11, 1), date(2026, 7, 31)), "MODIS_NRT": (date(2026, 8, 1), _today),
    "VIIRS_SNPP_SP": (date(2012, 1, 20), date(2026, 6, 30)), "VIIRS_SNPP_NRT": (date(2026, 7, 1), _today),
    "VIIRS_NOAA20_SP": (date(2018, 4, 1), date(2026, 6, 30)), "VIIRS_NOAA20_NRT": (date(2026, 7, 1), _today),
    "VIIRS_NOAA21_SP": (date(2023, 3, 1), date(2026, 6, 30)), "VIIRS_NOAA21_NRT": (date(2024, 1, 17), _today),
}


def get_availability(map_key: str) -> dict:
    """FIRMS data_availability tablosu: kaynak -> (ilk_tarih, son_tarih). Alinamazsa son bilinen degerler."""
    CACHE_DIR.mkdir(exist_ok=True)
    cp = CACHE_DIR / f"availability_{datetime.now(timezone.utc):%Y%m%d}.csv"
    text = None
    if cp.exists() and not REFRESH:
        text = cp.read_text()
    else:
        try:
            r = requests.get(f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{map_key}/ALL", timeout=30)
            r.raise_for_status()
            text = r.text
            cp.write_text(text)
        except Exception as e:
            print(f"  [bilgi] data_availability alinamadi ({e}); son bilinen degerler kullaniliyor")
    avail = {}
    if text:
        try:
            df = pd.read_csv(StringIO(text))
            for _, r_ in df.iterrows():
                avail[str(r_["data_id"])] = (date.fromisoformat(str(r_["min_date"])), date.fromisoformat(str(r_["max_date"])))
        except Exception:
            avail = {}
    return avail or dict(DEFAULT_AVAIL)


def plan_window(sensor: str, start, ndays: int, avail: dict):
    """
    Bir tarih penceresi icin hangi kaynak(lar)dan hangi gunlerin cekilecegini planlar:
    once arsiv (SP), arsivin bittigi yerden sonrasi NRT. Dondurur: [(kaynak_id, baslangic, gun_sayisi), ...].
    Sensor o tarihlerde calismiyorsa bos liste.
    """
    sp_id, nrt_id = SENSOR_IDS[sensor]
    end = start + timedelta(days=ndays - 1)
    out, cur = [], start
    sp = avail.get(sp_id)
    if sp and cur <= sp[1] and end >= sp[0]:
        st = max(cur, sp[0])
        en = min(end, sp[1])
        out.append((sp_id, st, (en - st).days + 1))
        cur = en + timedelta(days=1)
    nr = avail.get(nrt_id)
    if cur <= end and nr and end >= nr[0] and cur <= nr[1]:
        st = max(cur, nr[0])
        en = min(end, nr[1])
        out.append((nrt_id, st, (en - st).days + 1))
    return out


def _window_cache_path(source: str, start, days: int) -> Path:
    # Arsiv (SP) verisi degismez -> kalici onbellek; NRT gunluk yenilenir
    asof = "sp" if source.endswith("_SP") else datetime.now(timezone.utc).strftime("%Y%m%d")
    return CACHE_DIR / f"{source}_{AREA_BBOX.replace(',', '_')}_{start:%Y%m%d}_{days}d_asof{asof}.csv"


def _fetch_window(source: str, map_key: str, start, days: int):
    """[start .. start+days-1] (days <= 5). Dondurur: (DataFrame, onbellekten_mi)."""
    CACHE_DIR.mkdir(exist_ok=True)
    cp = _window_cache_path(source, start, days)
    if not REFRESH and cp.exists():
        try:
            return pd.read_csv(cp), True
        except pd.errors.EmptyDataError:
            return pd.DataFrame(), True
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
           f"{map_key}/{source}/{AREA_BBOX}/{days}/{start:%Y-%m-%d}")
    last_err = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=90)
            resp.raise_for_status()
            text = resp.text.strip()
            if not text:
                df = pd.DataFrame()
            elif "<html" in text[:200].lower() or text.lower().startswith("invalid"):
                raise RuntimeError(f"FIRMS hatasi ({source}, {start}): {text[:200]}")
            else:
                df = pd.read_csv(StringIO(text))
            df.to_csv(cp, index=False)
            if REQUEST_PAUSE > 0:
                time.sleep(REQUEST_PAUSE)
            return df, False
        except RuntimeError:
            raise
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            last_err = e
            code = e.response.status_code if getattr(e, "response", None) is not None else None
            if code == 400:
                raise
            if attempt < FETCH_RETRIES:
                wait = 2 ** attempt
                print(f"  [bilgi] {source} {start}: baglanti/sunucu hatasi, {wait} sn sonra tekrar ({attempt}/{FETCH_RETRIES})")
                time.sleep(wait)
    raise last_err


def fetch_sensor(sensor: str, map_key: str, start, end, avail: dict, verbose: bool = True) -> pd.DataFrame:
    """[start..end] araligini 5'er gunluk pencerelerle indirir; her pencerede SP/NRT'yi otomatik secer."""
    frames, n_calls, n_cached, cur = [], 0, 0, start
    while cur <= end:
        n = min(5, (end - cur).days + 1)
        for sid, st, nd in plan_window(sensor, cur, n, avail):
            df, cached = _fetch_window(sid, map_key, st, nd)
            frames.append(df)
            n_calls += 1
            n_cached += int(cached)
        cur += timedelta(days=n)
    frames = [f for f in frames if len(f)]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if verbose:
        if n_calls == 0:
            print(f"  {sensor}: bu tarih araliginda veri yok (sensor calismiyor / arsivde yok)")
        else:
            how = "onbellekten" if n_cached == n_calls else (f"{n_cached}/{n_calls} istek onbellekten" if n_cached else "indirildi")
            print(f"  {sensor}: {len(out)} tespit ({n_calls} istek, {how})")
    return out


def describe_plan(avail: dict, start, end):
    print(f"Kullanilacak kaynaklar ({start} .. {end}):")
    for sensor in SENSOR_IDS:
        pieces = plan_window(sensor, start, (end - start).days + 1, avail)
        txt = ", ".join(f"{sid} {st}..{st + timedelta(days=nd - 1)}" for sid, st, nd in pieces) or "bu aralikta veri yok"
        print(f"  {sensor:<13} {txt}")


# ----------------------------------------------------------------------------
# 2. STANDARDIZE
# ----------------------------------------------------------------------------
def _col(df, name):
    if name in df.columns:
        return df[name].to_numpy()
    return np.full(len(df), None, dtype=object)


def standardize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    cols = ["lat", "lon", "timestamp", "confidence", "frp",
            "daynight", "satellite", "scan", "track", "source"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    t = df["acq_time"].astype(int).astype(str).str.zfill(4)
    ts = pd.to_datetime(df["acq_date"].astype(str) + " " + t.str[:2] + ":" + t.str[2:],
                        format="%Y-%m-%d %H:%M")
    conf_map = {"l": 30, "n": 60, "h": 90}
    conf = df["confidence"].map(
        lambda v: conf_map.get(str(v).lower()[:1], np.nan) if isinstance(v, str)
        else float(v))
    return pd.DataFrame({
        "lat": df["latitude"].to_numpy(),
        "lon": df["longitude"].to_numpy(),
        "timestamp": ts.to_numpy(),
        "confidence": conf.to_numpy(),
        "frp": pd.to_numeric(df["frp"], errors="coerce").to_numpy() if "frp" in df else np.nan,
        "daynight": _col(df, "daynight"),
        "satellite": _col(df, "satellite"),
        "scan": _col(df, "scan"),
        "track": _col(df, "track"),
        "source": label,
    })


# ----------------------------------------------------------------------------
# 3. HARMONIZE
# ----------------------------------------------------------------------------
def neighbors(a: pd.DataFrame, b: pd.DataFrame, max_km: float, max_h: float):
    """
    a'nin her noktasi icin b'de max_km ve max_h icindeki TUM komsulari bulur.
    Dondurur: her a noktasi icin (b indeksleri, mesafe_km, zaman_farki_saat).
    """
    tree = BallTree(np.radians(b[["lat", "lon"]].to_numpy()), metric="haversine")
    idx, dist = tree.query_radius(np.radians(a[["lat", "lon"]].to_numpy()),
                                  r=max_km / EARTH_RADIUS_KM, return_distance=True)
    a_ts = a["timestamp"].to_numpy()
    b_ts = b["timestamp"].to_numpy()
    out = []
    for i in range(len(a)):
        ii = idx[i]
        if len(ii) == 0:
            out.append((ii, np.array([]), np.array([])))
            continue
        dt = np.abs((b_ts[ii] - a_ts[i]) / np.timedelta64(1, "h"))
        ok = dt <= max_h
        out.append((ii[ok], dist[i][ok] * EARTH_RADIUS_KM, dt[ok]))
    return out


def half_size(df: pd.DataFrame, default_km: float) -> np.ndarray:
    """Pikselin yaklasik yarim kenari (km): 0.5 * sqrt(scan * track)."""
    scan = pd.to_numeric(df["scan"], errors="coerce")
    track = pd.to_numeric(df["track"], errors="coerce")
    side = np.sqrt(scan * track).fillna(default_km * 2)
    return (0.5 * side).to_numpy(dtype=float)


def harmonize(modis: pd.DataFrame, viirs: pd.DataFrame):
    half_m = half_size(modis, 0.5)      # MODIS nominal 1 km
    half_v = half_size(viirs, 0.1875)   # VIIRS nominal 0.375 km

    # Genis aga bir kez at (3 km / 24 saat); duyarlilik tablosu bundan hesaplanir
    wide = neighbors(modis, viirs, max(SENS_DISTANCES), max(SENS_HOURS))

    rows = []
    for i, (ii, dd, dt) in enumerate(wide):
        m = modis.iloc[i]
        foot = half_m[i] + half_v[ii]                       # piksel ayak izi esigi
        sel = (dd <= foot) & (dt <= MAX_TIME_DELTA_HOURS)
        fixed = (dd <= MAX_DISTANCE_KM) & (dt <= MAX_TIME_DELTA_HOURS)
        row = {"modis_lat": m["lat"], "modis_lon": m["lon"],
               "modis_time": m["timestamp"], "modis_conf": m["confidence"],
               "modis_frp": m["frp"], "modis_daynight": m["daynight"],
               "modis_sat": m["satellite"], "modis_pixel_km": 2 * half_m[i],
               "viirs_lat": np.nan, "viirs_lon": np.nan, "viirs_time": pd.NaT,
               "viirs_source": None, "distance_km": np.nan,
               "time_delta_hours": np.nan, "n_viirs_in_window": int(sel.sum()),
               "matched": bool(sel.any()), "matched_fixed": bool(fixed.any())}
        if sel.any():
            k = np.argmin(np.where(sel, dd, np.inf))
            v = viirs.iloc[ii[k]]
            row.update({"viirs_lat": v["lat"], "viirs_lon": v["lon"],
                        "viirs_time": v["timestamp"], "viirs_source": v["source"],
                        "distance_km": dd[k], "time_delta_hours": dt[k]})
        rows.append(row)
    result = pd.DataFrame(rows)

    # Ters yon: her VIIRS noktasinin ayak izi icinde bir MODIS karsiligi var mi?
    rev = neighbors(viirs, modis, max(SENS_DISTANCES), MAX_TIME_DELTA_HOURS)
    viirs_has_modis = np.array(
        [bool(np.any(dd <= half_v[i] + half_m[ii])) for i, (ii, dd, _) in enumerate(rev)])
    return result, wide, viirs_has_modis


def cluster_points(df: pd.DataFrame, max_km: float, max_h: float):
    """
    Yakin (max_km) ve eszamanli (max_h) tespitleri zincirleme birlestirip tek
    'olay kumesi' yapar. Dondurur: her nokta icin olay etiketi.
    Amac: ayni yangini tekrar tekrar saymamak ve bagimsiz ornek sayisini gormek.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    X = np.radians(df[["lat", "lon"]].to_numpy())
    tree = BallTree(X, metric="haversine")
    idx = tree.query_radius(X, r=max_km / EARTH_RADIUS_KM)
    ts = df["timestamp"].to_numpy()
    rows, cols = [], []
    for i, ii in enumerate(idx):
        dt = np.abs((ts[ii] - ts[i]) / np.timedelta64(1, "h"))
        ok = ii[dt <= max_h]
        rows.extend([i] * len(ok))
        cols.extend(ok.tolist())
    n = len(df)
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return labels


def wilson(k: int, n: int, z: float = 1.96):
    """Oran icin %95 Wilson guven araligi (yuzde olarak)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (c - h), 100 * (c + h))


# ----------------------------------------------------------------------------
# 4. ANALIZ
# ----------------------------------------------------------------------------
def summarize(result, modis, viirs, viirs_has_modis, labels, labels_m):
    n_m, n_v = len(modis), len(viirs)
    matched = result[result["matched"]]
    n_fixed = int(result["matched_fixed"].sum())
    print("=" * 66)
    print(f"HARMONIZATION OZETI  (ayak izi esigi, zaman <= {MAX_TIME_DELTA_HOURS} saat)")
    print("=" * 66)
    print(f"MODIS tespit toplam:               {n_m}")
    print(f"VIIRS tespit toplam:               {n_v}  (VIIRS/MODIS = {n_v / max(n_m, 1):.1f}x)")
    print(f"VIIRS'te eslesen MODIS:            {len(matched)} ({len(matched) / max(n_m, 1) * 100:.1f}%)")
    print(f"  (karsilastirma: sabit {MAX_DISTANCE_KM} km esigi ile {n_fixed} = {n_fixed / max(n_m, 1) * 100:.1f}%)")
    print(f"VIIRS'te eslesmeyen MODIS:         {n_m - len(matched)}")
    if len(matched):
        print(f"Eslesen MODIS basina ort. VIIRS:   {matched['n_viirs_in_window'].mean():.1f} nokta")
        print(f"Ort. konum farki:                  {matched['distance_km'].mean():.3f} km")
        print(f"Ort. zaman farki:                  {matched['time_delta_hours'].mean():.2f} saat")
    print("-" * 66)
    ser = pd.Series(viirs_has_modis)
    ev_has = ser.groupby(labels).any()
    n_ev = len(ev_has)
    n_ev_with = int(ev_has.sum())
    print(f"VIIRS tespit -> olay kumesi:       {n_v} -> {n_ev}  ({n_v / max(n_ev, 1):.1f} tespit/olay)")
    print(f"MODIS karsiligi olan VIIRS olayi:  {n_ev_with} ({n_ev_with / max(n_ev, 1) * 100:.1f}%)")
    print(f"MODIS karsiligi OLMAYAN VIIRS olayi: {n_ev - n_ev_with} ({(n_ev - n_ev_with) / max(n_ev, 1) * 100:.1f}%)")
    print("  -> MODIS'in gormedigi kucuk/dusuk yogunluklu yangin adaylari (tekrarlar elendi).")
    print("-" * 66)
    ev_m = result["matched"].groupby(labels_m).any()
    k, n = int(ev_m.sum()), len(ev_m)
    lo, hi = wilson(k, n)
    print(f"MODIS tespit -> olay kumesi:       {n_m} -> {n}  ({n_m / max(n, 1):.1f} tespit/olay)")
    print(f"Olay duzeyinde VIIRS eslesmesi:    {k}/{n} = {k / max(n, 1) * 100:.1f}%  (%95 GA: {lo:.1f}-{hi:.1f})")
    print("  -> Bagimsiz yangin sayisi bu; nokta bazli yuzdeler kumelenme yuzunden iyimser/kararsiz olabilir.")
    print("=" * 66)


def sensitivity(wide, show: bool = True) -> pd.DataFrame:
    table = pd.DataFrame(index=SENS_DISTANCES, columns=SENS_HOURS, dtype=float)
    for d in SENS_DISTANCES:
        for h in SENS_HOURS:
            hits = [bool(np.any((dd <= d) & (dt <= h))) for _, dd, dt in wide]
            table.loc[d, h] = 100.0 * np.mean(hits)
    table.index.name = "mesafe_km \\ saat"
    if show:
        print("\nESIK DUYARLILIK TABLOSU - SABIT mesafe esigiyle eslesen MODIS yuzdesi")
        print(table.to_string(float_format=lambda x: f"{x:5.1f}"))
    return table


def breakdown(result):
    r = result.copy()
    try:
        r["frp_ceyrek"] = pd.qcut(r["modis_frp"], 4, duplicates="drop").astype(str)
    except Exception:
        r["frp_ceyrek"] = "n/a"
    g = r.groupby("frp_ceyrek", observed=True)["matched"].agg(["count", "mean"])
    g["mean"] = (g["mean"] * 100).round(1)
    g.columns = ["nokta", "eslesme_%"]
    print("\nMODIS FRP CEYREKLERINE GORE VIIRS ESLESMESI")
    print(g.to_string())


def sat_report(result):
    print("\nTERRA / AQUA AYRI RAPOR (VIIRS ile ~13:30 gecisi Aqua'ya yakin, Terra 3 saat once)")
    g = result.groupby("modis_sat").agg(
        nokta=("matched", "size"),
        eslesme_pct=("matched", lambda x: 100 * x.mean()),
        ort_zaman_farki_saat=("time_delta_hours", "mean"),
        ort_mesafe_km=("distance_km", "mean"),
        ort_VIIRS_sayisi=("n_viirs_in_window", "mean"))
    print(g.round(2).to_string())

    r = result.copy()
    r["saat_utc"] = pd.to_datetime(r["modis_time"]).dt.hour
    h = r.groupby("saat_utc")["matched"].agg(["count", "mean"])
    h["mean"] = (h["mean"] * 100).round(1)
    h.columns = ["nokta", "eslesme_%"]
    print("\nMODIS GECIS SAATINE (UTC) GORE ESLESME")
    print(h.to_string())


def region_report(result, min_points=10):
    r = result.copy()
    r["hucre"] = (r["modis_lat"].round().astype(int).astype(str) + "N, "
                  + r["modis_lon"].round().astype(int).astype(str) + "E")
    r["aqua"] = r["modis_sat"].astype(str).eq("Aqua")
    g = r.groupby("hucre").agg(nokta=("matched", "size"),
                               eslesme_pct=("matched", lambda x: 100 * x.mean()),
                               aqua_payi_pct=("aqua", lambda x: 100 * x.mean()),
                               ort_zaman_farki_saat=("time_delta_hours", "mean"))
    g = g[g["nokta"] >= min_points].sort_values("nokta", ascending=False).head(10)
    print("\nBOLGE (1 derece hucre) - en cok noktali 10 hucre")
    print(g.round(1).to_string())


def window_table(modis, viirs, wide):
    """Terra/Aqua icin: ayak izi cakisan VIIRS var mi, zaman penceresi 1/3/6/12 saat."""
    half_m = half_size(modis, 0.5)
    half_v = half_size(viirs, 0.1875)
    hits = {w: [] for w in WINDOWS}
    any24 = []
    for i, (ii, dd, dt) in enumerate(wide):
        ov = dd <= (half_m[i] + half_v[ii])
        for w in WINDOWS:
            hits[w].append(bool(np.any(ov & (dt <= w))))
        any24.append(bool(np.any(ov)))      # wide zaten <=24 saat
    df = pd.DataFrame({f"<={w}sa": hits[w] for w in WINDOWS})
    df["<=24sa"] = any24
    df["sat"] = modis["satellite"].to_numpy()

    tab = df.groupby("sat").mean(numeric_only=True) * 100
    tab.insert(0, "nokta", df.groupby("sat").size())
    tab.loc["TUMU"] = [len(df)] + list(df.drop(columns="sat").mean() * 100)
    tab["nokta"] = tab["nokta"].astype(int)
    print("\nTERRA / AQUA x ZAMAN PENCERESI - ayak izi cakisan VIIRS bulunan MODIS yuzdesi")
    print(tab.round(1).to_string())

    print("\n%95 guven araliklari (nokta bazli):")
    for sat in [x for x in tab.index if x != "TUMU"]:
        sub = df[df["sat"] == sat]
        for w in (1, 6):
            kk = int(sub[f"<={w}sa"].sum())
            lo, hi = wilson(kk, len(sub))
            print(f"  {sat:<6} <={w:>2} saat: {kk / len(sub) * 100:5.1f}%  (GA {lo:.1f}-{hi:.1f}, n={len(sub)})")
    return tab


def daily_report(result, viirs):
    d = pd.to_datetime(result["modis_time"]).dt.date
    g = result.groupby(d)["matched"].agg(["count", "mean"])
    g["mean"] = (g["mean"] * 100).round(1)
    g.columns = ["MODIS_nokta", "eslesme_%"]
    vd = pd.to_datetime(viirs["timestamp"]).dt.date.value_counts().sort_index()
    g["VIIRS_nokta"] = vd.reindex(g.index).fillna(0).astype(int)
    print("\nGUNLUK KIRILIM (tespit tarihi, UTC; ilk/son gun kismi olabilir)")
    print(g.to_string())


def _hhmm(x: float) -> str:
    x = x % 24
    h = int(x)
    m = int(round((x - h) * 60))
    if m == 60:
        h, m = (h + 1) % 24, 0
    return f"{h:02d}:{m:02d}"


def _stack(modis, viirs):
    """MODIS + VIIRS tek tabloda; grup = Terra/Aqua/VIIRS_SNPP/..."""
    allp = pd.concat([modis.assign(grup=modis["satellite"].astype(str)),
                      viirs.assign(grup=viirs["source"].astype(str))],
                     ignore_index=True)
    half = np.concatenate([half_size(modis, 0.5), half_size(viirs, 0.1875)])
    return allp, half


def pass_time_table(modis, viirs):
    """Her uydunun gecis saati: yaklasik yerel gunes saati = UTC saati + boylam/15."""
    allp, _ = _stack(modis, viirs)
    ts = pd.to_datetime(allp["timestamp"])
    utc_h = ts.dt.hour + ts.dt.minute / 60
    local = ((utc_h + allp["lon"] / 15.0) % 24).to_numpy()
    is_night = allp["daynight"].astype(str).str.upper().eq("N").to_numpy()
    # gece gecisleri gece yarisini asabilir: 12'den kucukleri +24 yaparak sureklilik sagla
    local = np.where(is_night & (local < 12), local + 24, local)
    allp = allp.assign(yerel=local, gece=np.where(is_night, "N", "D"))
    rows = []
    for (g, dn), sub in allp.groupby(["grup", "gece"]):
        if len(sub) < 20:
            continue
        q1, q2, q3 = sub["yerel"].quantile([0.25, 0.5, 0.75])
        rows.append({"uydu": g, "D/N": dn, "nokta": len(sub),
                     "medyan_yerel_saat": _hhmm(q2), "Q1-Q3": f"{_hhmm(q1)}-{_hhmm(q3)}"})
    tab = pd.DataFrame(rows)
    print("\nGECIS SAATLERI - yaklasik yerel gunes saati")
    print("(nominal: Terra 10:30/22:30, Aqua-SNPP-NOAA 13:30/01:30)")
    print(tab.to_string(index=False))
    return tab


def agreement_matrix(modis, viirs, max_h: float = 6.0, min_points: int = 30):
    """
    Satir A'daki tespitlerin yuzde kaci, sutun B'de <= max_h saat icinde ayak izi cakisan
    bir tespite sahip? + cakisan ciftlerde medyan en kucuk zaman farki.
    VIIRS<->VIIRS = ayni sensor, farkli uydu/zaman (referans). "VIIRS(diger)" = A'nin kendisi
    haric tum VIIRS uydulari birlesik (MODIS->VIIRS(hepsi) ile adil kiyas icin).
    """
    allp, half = _stack(modis, viirs)
    order = ["Terra", "Aqua", "VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21"]
    groups = [g for g in order if (allp["grup"] == g).sum() >= min_points]
    cols = groups + ["VIIRS(diger)"]
    pct = pd.DataFrame(np.nan, index=groups, columns=cols)
    lag = pd.DataFrame(np.nan, index=groups, columns=cols)
    grp = allp["grup"].astype(str).to_numpy()
    is_v = np.array([g.startswith("VIIRS") for g in grp])

    def run(ia, ib):
        A, B = allp.iloc[ia], allp.iloc[ib]
        ha, hb = half[ia], half[ib]
        nb = neighbors(A, B, 3.0, max_h)
        hits, lags = 0, []
        for i, (ii, dd, dt) in enumerate(nb):
            ov = dd <= (ha[i] + hb[ii])
            if np.any(ov):
                hits += 1
                lags.append(float(dt[ov].min()))
        return 100.0 * hits / len(A), (float(np.median(lags)) if lags else np.nan)

    for a_ in groups:
        ia = np.flatnonzero(grp == a_)
        for b_ in groups:
            if a_ == b_:
                continue
            pct.loc[a_, b_], lag.loc[a_, b_] = run(ia, np.flatnonzero(grp == b_))
        ib = np.flatnonzero(is_v & (grp != a_))
        if len(ib):
            pct.loc[a_, "VIIRS(diger)"], lag.loc[a_, "VIIRS(diger)"] = run(ia, ib)
    print(f"\nUYUM MATRISI - satirdaki tespitlerin yuzde kaci sutundaki uyduda <= {max_h:g} saat icinde karsilik buluyor")
    print(pct.round(1).to_string(na_rep="-"))
    print("\nCAKISAN CIFTLERDE MEDYAN EN KUCUK ZAMAN FARKI (saat)")
    print(lag.round(2).to_string(na_rep="-"))
    return pct, lag


def static_counts(modis, viirs):
    """Her tespit icin ayni yerde (VIIRS 0.5 km / MODIS 1 km) kac FARKLI gunde tespit oldugunu sayar."""
    ts_all = pd.concat([pd.to_datetime(modis["timestamp"]), pd.to_datetime(viirs["timestamp"])])
    t0 = ts_all.min().normalize()
    n_days = int((ts_all.max().normalize() - t0).days) + 1
    dv = (pd.to_datetime(viirs["timestamp"]).dt.normalize() - t0).dt.days.to_numpy()
    dm = (pd.to_datetime(modis["timestamp"]).dt.normalize() - t0).dt.days.to_numpy()
    Xv = np.radians(viirs[["lat", "lon"]].to_numpy())
    Xm = np.radians(modis[["lat", "lon"]].to_numpy())
    tree_v = BallTree(Xv, metric="haversine")
    tree_m = BallTree(Xm, metric="haversine")
    r_v = STATIC_RADIUS_KM / EARTH_RADIUS_KM
    r_m = 1.0 / EARTH_RADIUS_KM            # MODIS pikseli ~1 km
    idx_vv = tree_v.query_radius(Xv, r=r_v)
    nd_v = np.array([len(np.unique(dv[ii])) for ii in idx_vv])
    idx_mv = tree_v.query_radius(Xm, r=r_m)
    idx_mm = tree_m.query_radius(Xm, r=r_m)
    nd_m = np.array([len(np.unique(np.concatenate([dv[a_], dm[b_]]))) for a_, b_ in zip(idx_mv, idx_mm)])
    return nd_m, nd_v, n_days


def flag_static(modis, viirs):
    """Kalici kaynak adaylari: ayni yerde >= min_days farkli gunde tekrar. Dondurur 6'li (bayraklar + sayimlar)."""
    nd_m, nd_v, n_days = static_counts(modis, viirs)
    min_days = max(STATIC_MIN_DAYS, int(np.ceil(STATIC_FRACTION * n_days)))
    if n_days < STATIC_MIN_DAYS:
        return np.zeros(len(modis), bool), np.zeros(len(viirs), bool), n_days, min_days, nd_m, nd_v
    return nd_m >= min_days, nd_v >= min_days, n_days, min_days, nd_m, nd_v


def report_static(modis, viirs, static_m, static_v, n_days, min_days, nd_m, nd_v):
    print("=" * 66)
    print(f"KALICI KAYNAK ADAYLARI (ayni yerde >= {min_days} farkli gunde tespit; pencere {n_days} gun)")
    print("=" * 66)
    if n_days < STATIC_MIN_DAYS:
        print("Veri penceresi cok kisa, kalici kaynak taramasi atlandi.")
        return
    res, _, _ = harmonize(modis, viirs)
    m = res["matched"].to_numpy()
    print(f"VIIRS: {int(static_v.sum())}/{len(viirs)} (%{100 * static_v.mean():.1f}) kalici kaynak adayi")
    for name, mask in [("kalici", static_m), ("diger", ~static_m)]:
        n = int(mask.sum())
        rate = 100 * m[mask].mean() if n else float("nan")
        print(f"MODIS {name:<7}: n={n:4d}   VIIRS eslesmesi %{rate:.1f}")
    if static_m.any():
        r = modis[static_m]
        cells = (r["lat"].round().astype(int).astype(str) + "N, "
                 + r["lon"].round().astype(int).astype(str) + "E").value_counts().head(5)
        print("Kalici MODIS adaylarinin en yogun 1 derece hucreleri:")
        print(cells.to_string())
    rows = []
    for k in [2, 3, 5, 8, 12, 15, 20, 25]:
        if k > n_days:
            continue
        sm, sv = nd_m >= k, nd_v >= k
        rows.append({"esik_gun": k, "MODIS_kalici": int(sm.sum()), "MODIS_%": 100 * sm.mean(), "VIIRS_%": 100 * sv.mean(),
                     "kalici_MODIS_esl_%": 100 * m[sm].mean() if sm.any() else np.nan,
                     "diger_MODIS_esl_%": 100 * m[~sm].mean() if (~sm).any() else np.nan})
    print("\nESIK DUYARLILIGI - ayni yerde kac farkli gunde tekrar ederse 'kalici' sayilsin?")
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    print("(Not: haftalarca suren buyuk yanginlar da yanlislikla isaretlenebilir.)")


def trim_incomplete(modis, viirs):
    """
    Bir kaynak digerlerinden geride ise (ornegin VIIRS_SNPP_NRT'nin son tarihi bir gun once) ya da
    bugun yarim ise, eksik olabilecek son gunleri analizden cikarir.
    """
    parts = [("MODIS", modis)] + [(g, viirs[viirs["source"] == g]) for g in sorted(viirs["source"].unique())]
    last = {n: pd.to_datetime(d["timestamp"]).max().normalize() for n, d in parts if len(d)}
    cutoff = min(last.values())
    if max(last.values()) > cutoff:
        cutoff -= pd.Timedelta(days=1)          # geride kalan kaynagin son gunu da eksik olabilir
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    cutoff = min(cutoff, today - pd.Timedelta(days=1))
    end = cutoff + pd.Timedelta(days=1)
    mk = (pd.to_datetime(modis["timestamp"]) < end).to_numpy()
    vk = (pd.to_datetime(viirs["timestamp"]) < end).to_numpy()
    n_drop = int((~mk).sum() + (~vk).sum())
    info = ", ".join(f"{n}:{d.date()}" for n, d in last.items())
    print(f"Son tam gun: {cutoff.date()} (kaynaklarin son tarihleri: {info}); eksik olabilecek {n_drop} tespit atildi.")
    return modis[mk].reset_index(drop=True), viirs[vk].reset_index(drop=True)


def reverse_hits(viirs, modis_sub):
    """Her VIIRS tespiti icin modis_sub icinde ayak izi cakisan MODIS tespiti var mi?"""
    if len(modis_sub) == 0:
        return np.zeros(len(viirs), bool)
    modis_sub = modis_sub.reset_index(drop=True)
    half_v = half_size(viirs, 0.1875)
    half_m = half_size(modis_sub, 0.5)
    rev = neighbors(viirs, modis_sub, max(SENS_DISTANCES), MAX_TIME_DELTA_HOURS)
    return np.array([bool(np.any(dd <= half_v[i] + half_m[ii])) for i, (ii, dd, _) in enumerate(rev)])


def efficiency_table(viirs, labels, has_any, has_aqua, has_terra):
    """VIIRS olay FRP'si (tek gecis tahmini) arttikca MODIS'in o olayi gorme orani."""
    d = pd.DataFrame({"ev": labels,
                      "frp": pd.to_numeric(viirs["frp"], errors="coerce").to_numpy(float),
                      "src": viirs["source"].to_numpy()})
    ev = d.groupby("ev").agg(frp_sum=("frp", "sum"), nsat=("src", "nunique"))
    ev["frp"] = ev["frp_sum"] / ev["nsat"]          # 3 uydunun tekrarini bolerek tek gecis FRP'si
    ev["any"] = pd.Series(has_any).groupby(labels).any()
    ev["aqua"] = pd.Series(has_aqua).groupby(labels).any()
    ev["terra"] = pd.Series(has_terra).groupby(labels).any()
    edges = [0, 2, 5, 10, 20, 50, np.inf]
    names = ["<2", "2-5", "5-10", "10-20", "20-50", ">=50"]
    ev["bin"] = pd.cut(ev["frp"], edges, labels=names, right=False)
    rows = []
    for b_, g in ev.groupby("bin", observed=True):
        k, n = int(g["any"].sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append({"olay_FRP_MW": b_, "olay": n, "MODIS_herhangi_%": 100 * k / n,
                     "GA95": f"{lo:.0f}-{hi:.0f}", "Aqua_%": 100 * g["aqua"].mean(),
                     "Terra_%": 100 * g["terra"].mean()})
    print("\nTESPIT VERIMLILIGI - VIIRS olayinin FRP'sine gore MODIS'in o olayi gorme orani")
    print("(olay FRP'si = olaydaki VIIRS FRP toplami / olayi gorup uydu sayisi)")
    print(pd.DataFrame(rows).round(1).to_string(index=False))



def frp_calibration(modis, viirs, wide):
    """Eslesen MODIS FRP ile ayni VIIRS gecisinin cakisan piksel FRP toplamini karsilastirir."""
    half_m = half_size(modis, 0.5)
    half_v = half_size(viirs, 0.1875)
    v_frp = pd.to_numeric(viirs["frp"], errors="coerce").to_numpy(float)
    v_src = viirs["source"].to_numpy()
    v_ts = viirs["timestamp"].to_numpy()
    m_frp = pd.to_numeric(modis["frp"], errors="coerce").to_numpy(float)
    m_sat = modis["satellite"].astype(str).to_numpy()
    rows = []
    for i, (ii, dd, dt) in enumerate(wide):
        if len(ii) == 0:
            continue
        ov = (dd <= half_m[i] + half_v[ii]) & (dt <= MAX_TIME_DELTA_HOURS)
        if not ov.any():
            continue
        cand = np.flatnonzero(ov)
        j0 = cand[np.argmin(dt[cand])]
        src0, t0 = v_src[ii[j0]], v_ts[ii[j0]]
        same = [c for c in cand if v_src[ii[c]] == src0
                and abs((v_ts[ii[c]] - t0) / np.timedelta64(1, "h")) <= 0.25]
        vs = float(np.nansum(v_frp[ii[same]]))
        if np.isfinite(m_frp[i]) and m_frp[i] > 0 and vs > 0:
            rows.append({"modis_frp": m_frp[i], "viirs_frp_toplam": vs, "viirs_piksel": len(same),
                         "zaman_farki_saat": float(dt[j0]), "modis_uydu": m_sat[i], "viirs_uydu": src0})
    df = pd.DataFrame(rows)
    print("\nFRP KALIBRASYONU - eslesen MODIS FRP / ayni VIIRS gecisinin cakisan piksellerinin FRP toplami")
    if len(df) < 10:
        print("  yeterli eslesen cift yok.")
        return df
    df["oran"] = df["modis_frp"] / df["viirs_frp_toplam"]

    def line(label, g):
        q1, q2, q3 = g["oran"].quantile([0.25, 0.5, 0.75])
        print(f"  {label:<18} n={len(g):4d}   medyan oran {q2:5.2f}  (IQR {q1:.2f}-{q3:.2f})")

    line("tum eslesenler", df)
    for sat_, g in df.groupby("modis_uydu"):
        line(f"MODIS {sat_}", g)
    df["lag_bin"] = pd.cut(df["zaman_farki_saat"], [-0.01, 2, 4, 6.01], labels=["0-2sa", "2-4sa", "4-6sa"])
    for lb, g in df.groupby("lag_bin", observed=True):
        line(f"zaman farki {lb}", g)
    if len(df) >= 20:
        slope, icpt = np.polyfit(np.log(df["viirs_frp_toplam"]), np.log(df["modis_frp"]), 1)
        r2 = np.corrcoef(np.log(df["viirs_frp_toplam"]), np.log(df["modis_frp"]))[0, 1] ** 2
        print(f"  log-log uyum: ln(MODIS) = {icpt:.2f} + {slope:.2f} * ln(VIIRS toplam)   R2={r2:.2f}")
    print("  (Not: sadece IKISI de tespit eden ciftler; dusuk FRP'de secilim yanliligi vardir.)")
    df.to_csv(OUT_DIR / "frp_pairs.csv", index=False)
    return df


def _grid(allp, cell: float):
    """Hucre x gun tablolari (sayi, FRP). Aktif hucreler (pencerede en az 1 tespit) x TUM gunler, sifirlar dahil."""
    a_ = allp.assign(ci=np.floor(allp["lat"] / cell).astype(int), cj=np.floor(allp["lon"] / cell).astype(int))
    g = a_.groupby(["ci", "cj", "gun", "grup"]).agg(n=("lat", "size"), frp=("frp_n", "sum")).reset_index()
    cnt = g.pivot_table(index=["ci", "cj", "gun"], columns="grup", values="n", fill_value=0)
    frp = g.pivot_table(index=["ci", "cj", "gun"], columns="grup", values="frp", fill_value=0.0)
    cells = sorted(set(zip(g["ci"], g["cj"])))
    days = sorted(g["gun"].unique())
    full = pd.MultiIndex.from_tuples([(i, j, d) for (i, j) in cells for d in days], names=["ci", "cj", "gun"])
    cnt = cnt.reindex(full, fill_value=0)
    frp = frp.reindex(full, fill_value=0.0)
    for c_ in ["Terra", "Aqua"]:
        for tab_ in (cnt, frp):
            if c_ not in tab_.columns:
                tab_[c_] = 0
    v_cols = [c_ for c_ in cnt.columns if str(c_).startswith("VIIRS")]
    for tab_ in (cnt, frp):
        tab_["VIIRS_ort"] = tab_[v_cols].mean(axis=1)
    cnt["VIIRS_max"] = cnt[v_cols].max(axis=1)
    return cnt, frp


def _anom(x):
    """Hucre ortalamasi cikarilmis log1p sayi: 'nerede' bilgisini siler, sadece ayni hucrede gunler arasi
    degisimi (NE ZAMAN) birakir."""
    lx = np.log1p(x)
    return lx - lx.groupby(level=["ci", "cj"]).transform("mean")


def _corr_table(cnt, frp, title):
    pairs = [("Aqua", "VIIRS_ort"), ("Terra", "VIIRS_ort"), ("Aqua", "VIIRS_SNPP"),
             ("Terra", "Aqua"), ("VIIRS_SNPP", "VIIRS_NOAA21"), ("VIIRS_SNPP", "VIIRS_NOAA20")]
    rows = []
    for a_, b_ in pairs:
        if a_ not in cnt.columns or b_ not in cnt.columns:
            continue
        x, y = cnt[a_], cnt[b_]
        rows.append({"cift": f"{a_} ~ {b_}", "Spearman_sayi": x.corr(y, method="spearman"),
                     "Pearson_log1p_sayi": np.log1p(x).corr(np.log1p(y)),
                     "Spearman_FRP": frp[a_].corr(frp[b_], method="spearman"),
                     "Zamansal_log1p": _anom(x).corr(_anom(y))})
    print(f"\n{title}  (hucre-gun: {len(cnt)})")
    print("(Zamansal_log1p = hucre ortalamasi cikarildiktan sonra korelasyon; konum etkisi disarida)")
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def thinned_control(allp, reps: int = 15, seed: int = 0):
    """
    MODIS~VIIRS korelasyonu dusuk cikarsa iki aday sebep var: (1) MODIS'te nokta az, sayilar gurultulu,
    (2) gercek zaman/sensor farki. Kontrol: VIIRS_SNPP'yi MODIS'in seyrek yogunluguna indir, VIIRS_NOAA20
    ile korele et. gercek/kontrol ~1 ise MODIS sadece "seyrek" bir VIIRS kadar tutarli; <1 ise ek uyumsuzluk var.
    """
    grup = allp["grup"].to_numpy()
    n_snpp = int((grup == "VIIRS_SNPP").sum())
    if n_snpp == 0 or int((grup == "VIIRS_NOAA20").sum()) == 0:
        return
    rng = np.random.default_rng(seed)
    rows = []
    for name in ["Aqua", "Terra"]:
        n_ref = int((grup == name).sum())
        if n_ref < 30:
            continue
        p = min(1.0, n_ref / n_snpp)
        for cell in (GRID_DEG, 1.0):
            c_real, _ = _grid(allp, cell)
            real = c_real[name].corr(c_real["VIIRS_SNPP"], method="spearman")
            vals = []
            for _ in range(reps):
                keep = (grup != "VIIRS_SNPP") | (rng.random(len(allp)) < p)
                c, _f = _grid(allp[keep], cell)
                vals.append(c["VIIRS_SNPP"].corr(c["VIIRS_NOAA20"], method="spearman"))
            m_, sd_ = float(np.mean(vals)), float(np.std(vals))
            rows.append({"MODIS": name, "hucre_derece": f"{cell:g}", "nokta_orani": round(p, 3),
                         "gercek_MODIS~SNPP": real, "kontrol_seyrekSNPP~NOAA20": m_, "kontrol_sd": sd_,
                         "gercek/kontrol": real / m_ if m_ else np.nan})
    if rows:
        print("\nSEYRELTILMIS AYNI-SENSOR KONTROLU (Spearman, sayi)")
        print("(VIIRS_SNPP, MODIS'in nokta yogunluguna rastgele seyreltilip VIIRS_NOAA20 ile korele edildi; "
              f"{reps} tekrar)")
        print(pd.DataFrame(rows).round(2).to_string(index=False))


def grid_comparison(modis, viirs, cell: float = GRID_DEG):
    """
    Nokta eslestirmesi yerine hucre x gun toplamlari (yangin aktivitesi zaman serileri genelde
    boyle kiyaslanir). Sorular: iki sensor NEREDE ve NE KADAR yangin oldugunda hemfikir?
    VIIRS aktivitesi artinca MODIS'in o hucre-gunde en az bir tespit yapma olasiligi nedir?
    """
    allp = pd.concat([modis.assign(grup=modis["satellite"].astype(str)),
                      viirs.assign(grup=viirs["source"].astype(str))], ignore_index=True)
    allp["gun"] = pd.to_datetime(allp["timestamp"]).dt.date.astype(str)
    allp["frp_n"] = pd.to_numeric(allp["frp"], errors="coerce").fillna(0.0)

    cnt, frp = _grid(allp, cell)
    has_m = (cnt["Terra"] + cnt["Aqua"]) > 0
    has_v = cnt["VIIRS_max"] > 0
    print(f"\nIZGARA KARSILASTIRMASI ({cell:g} derece hucre x gun, aktif hucreler x tum gunler)")
    print(f"hucre-gun: {len(cnt)} | MODIS var: {int(has_m.sum())} | VIIRS var: {int(has_v.sum())} | ikisi de: {int((has_m & has_v).sum())}")
    print(f"MODIS'li hucre-gunlerin %{100 * (has_m & has_v).sum() / max(has_m.sum(), 1):.1f}'inde VIIRS de var; "
          f"VIIRS'li hucre-gunlerin %{100 * (has_m & has_v).sum() / max(has_v.sum(), 1):.1f}'inde MODIS de var")

    _corr_table(cnt, frp, f"KORELASYON - {cell:g} derece")
    cnt1, frp1 = _grid(allp, 1.0)
    _corr_table(cnt1, frp1, "KORELASYON - 1 derece (daha kaba, daha yogun sayilar)")
    thinned_control(allp)

    cc = cnt[cnt["VIIRS_max"] > 0].copy()
    cc["bin"] = pd.cut(cc["VIIRS_max"], [1, 2, 4, 8, 16, np.inf], labels=["1", "2-3", "4-7", "8-15", ">=16"], right=False)
    cc["m_any"] = ((cc["Terra"] + cc["Aqua"]) > 0)
    t = cc.groupby("bin", observed=True).agg(
        hucre_gun=("VIIRS_max", "size"),
        P_Aqua=("Aqua", lambda x: 100 * (x > 0).mean()),
        P_Terra=("Terra", lambda x: 100 * (x > 0).mean()),
        P_MODIS_herhangi=("m_any", lambda x: 100 * x.mean()),
        ort_Aqua_sayi=("Aqua", "mean"), ort_Terra_sayi=("Terra", "mean"))
    print(f"\nVIIRS AKTIVITESINE GORE MODIS'IN O HUCRE-GUNDE (%{cell:g} derece) EN AZ 1 TESPIT YAPMA OLASILIGI (%)")
    print("(VIIRS aktivitesi = o hucre-gunde tek bir VIIRS uydusunun gordugu en yuksek tespit sayisi)")
    print(t.round(1).to_string())

    days = cnt.groupby(level="gun")[["Terra", "Aqua", "VIIRS_ort"]].sum()
    fr = frp.groupby(level="gun")[["Terra", "Aqua", "VIIRS_ort"]].sum()
    ratio_cols = ["Aqua/VIIRS_sayi", "Aqua/VIIRS_FRP", "Terra/VIIRS_FRP"]

    def build(sc, sf, extra=None):
        d = pd.DataFrame({"sayi_Terra": sc["Terra"], "sayi_Aqua": sc["Aqua"], "sayi_VIIRS": sc["VIIRS_ort"].round(0),
                          "FRP_Terra": sf["Terra"].round(0), "FRP_Aqua": sf["Aqua"].round(0), "FRP_VIIRS": sf["VIIRS_ort"].round(0)})
        if extra is not None:
            d.insert(0, "gun_sayisi", extra)
        d["Aqua/VIIRS_sayi"] = sc["Aqua"] / sc["VIIRS_ort"]
        d["Aqua/VIIRS_FRP"] = sf["Aqua"] / sf["VIIRS_ort"]
        d["Terra/VIIRS_FRP"] = sf["Terra"] / sf["VIIRS_ort"]
        return d

    if len(days) > 10:
        wk = pd.Series(np.arange(len(days)) // 7, index=days.index)
        n_in = wk.value_counts().sort_index()
        d = build(days.groupby(wk).sum(), fr.groupby(wk).sum(), n_in.to_numpy())
        d.index = [days.index[wk.to_numpy() == w][0] for w in d.index]
        d.index.name = "hafta_baslangic"
        core = d[d["gun_sayisi"] == 7]
        print("\nHAFTALIK TOPLAMLAR (VIIRS = mevcut VIIRS uydularinin ortalamasi) ve MODIS/VIIRS orani")
        print(d.round(2).to_string())
        unit = "tam hafta"
    else:
        d = build(days, fr)
        core = d.iloc[1:-1] if len(d) >= 4 else d
        print("\nGUNLUK TOPLAMLAR (VIIRS = mevcut VIIRS uydularinin ortalamasi) ve MODIS/VIIRS orani")
        print(d.round(2).to_string())
        unit = "ilk/son gun haric"
    for col in ratio_cols:
        mu, sd = core[col].mean(), core[col].std()
        cv = f"  CV {100 * sd / mu:.0f}%" if mu else ""
        print(f"  {col:<16} {unit} (n={len(core)}): ort {mu:.2f}  sd {sd:.2f}{cv}")
    cnt.add_prefix("n_").join(frp.add_prefix("frp_")).to_csv(OUT_DIR / "grid_cell_day.csv")
    return cnt, frp


def daily_quality(modis, viirs, result):
    """
    Gun x uydu tespit sayilari. Bir uydunun diger uydulara gore payi ani dususe gecerse (veri bosluğu/kesinti
    suphesi) o gun bayraklanir; dusuk aktivite gunleri (VIIRS az) bayraklanmaz.
    Dondurur: bayrakli gunlerin kumesi ("YYYY-MM-DD").
    """
    groups = ["Terra", "Aqua", "VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21"]
    allp = pd.concat([modis.assign(grup=modis["satellite"].astype(str)),
                      viirs.assign(grup=viirs["source"].astype(str))], ignore_index=True)
    allp["gun"] = pd.to_datetime(allp["timestamp"]).dt.date.astype(str)
    tab = pd.crosstab(allp["gun"], allp["grup"]).reindex(columns=groups, fill_value=0)
    full = pd.date_range(pd.to_datetime(tab.index.min()), pd.to_datetime(tab.index.max())).strftime("%Y-%m-%d")
    tab = tab.reindex(full, fill_value=0)
    v_cols = [c_ for c_ in groups if c_.startswith("VIIRS")]
    v_ort = tab[v_cols].mean(axis=1)
    ok_day = v_ort >= QC_MIN_VIIRS
    flags = pd.Series("", index=tab.index)
    for g in ["Terra", "Aqua"]:
        share = tab[g] / v_ort.replace(0, np.nan)
        med = share[ok_day].median()
        bad = ok_day & (share < QC_RATIO * med)
        flags[bad] = flags[bad] + g + " "
    for g in v_cols:
        o = tab[[c_ for c_ in v_cols if c_ != g]].mean(axis=1)
        share = tab[g] / o.replace(0, np.nan)
        bad = (o >= QC_MIN_VIIRS) & (share < 0.3)
        flags[bad] = flags[bad] + g.replace("VIIRS_", "") + " "
    r = result.copy()
    r["gun"] = pd.to_datetime(r["modis_time"]).dt.date.astype(str)
    mt = r.groupby(["gun", r["modis_sat"].astype(str)])["matched"].mean().unstack() * 100
    out = tab.rename(columns={"VIIRS_SNPP": "SNPP", "VIIRS_NOAA20": "N20", "VIIRS_NOAA21": "N21"})
    for g in ["Terra", "Aqua"]:
        out[f"{g}_esl_%"] = mt[g].reindex(out.index).round(0) if g in mt.columns else np.nan
    out["bayrak"] = flags.str.strip()
    print("\nGUNLUK VERI KALITESI - gun x uydu tespit sayisi, MODIS uydu bazinda VIIRS eslesmesi (%), bayraklar")
    print(f"(bayrak: VIIRS ort. >= {QC_MIN_VIIRS} olan gunde bir uydunun payi medyanin %{int(QC_RATIO * 100)}'inden azsa)")
    print(out.to_string())
    flagged = set(out.index[out["bayrak"] != ""])
    if flagged:
        m_f = r.loc[r["gun"].isin(flagged), "matched"]
        m_n = r.loc[~r["gun"].isin(flagged), "matched"]
        print(f"Bayrakli {len(flagged)} gun: nokta eslesmesi %{100 * m_f.mean() if len(m_f) else float('nan'):.1f} (n={len(m_f)}); "
              f"diger gunler: %{100 * m_n.mean() if len(m_n) else float('nan'):.1f} (n={len(m_n)})")
    else:
        print("Bayrakli gun yok.")
    return flagged


def _pdev(y, p):
    """Ortalama Poisson sapmasi."""
    p = np.clip(p, 1e-9, None)
    ysafe = np.where(y > 0, y, 1.0)
    t = np.where(y > 0, y * np.log(ysafe / p), 0.0) - (y - p)
    return 2.0 * float(np.mean(t))


def transfer_model(modis, viirs, skip_days=(), cell: float = GRID_DEG, train_frac: float = 2 / 3):
    """
    VIIRS aktivitesinden (hucre-gun sayi + FRP, uydu ortalamasi) MODIS Aqua / Terra hucre-gun sayisini tahmin eder.
    Zaman ayrimli dogrulama: ilk train_frac kadar gun egitim, sonrasi test. Sabit carpan (tek oran) ile kiyaslanir.
    """
    import warnings
    from sklearn.linear_model import PoissonRegressor
    allp = pd.concat([modis.assign(grup=modis["satellite"].astype(str)),
                      viirs.assign(grup=viirs["source"].astype(str))], ignore_index=True)
    allp["gun"] = pd.to_datetime(allp["timestamp"]).dt.date.astype(str)
    allp["frp_n"] = pd.to_numeric(allp["frp"], errors="coerce").fillna(0.0)
    allp = allp[~allp["gun"].isin(set(skip_days))]
    cnt, frp = _grid(allp, cell)
    gun = cnt.index.get_level_values("gun")
    ds = sorted(gun.unique())
    print(f"\nVIIRS -> MODIS DONUSUM MODELI ({cell:g} derece hucre-gun; bayrakli {len(set(skip_days))} gun cikarildi)")
    if len(ds) < 6:
        print("  yeterli gun yok (en az 6 gun gerekir).")
        return
    n_tr = max(1, int(len(ds) * train_frac))
    tr_days = set(ds[:n_tr])
    is_tr = np.array([g in tr_days for g in gun])
    print(f"Egitim: {ds[0]} .. {ds[n_tr - 1]} ({n_tr} gun) | Test: {ds[n_tr]} .. {ds[-1]} ({len(ds) - n_tr} gun)")
    print("Hedef: o hucre-gunde MODIS tespit sayisi. Girdi: ayni hucre-gunde VIIRS sayi ve FRP (uydu ortalamasi).")
    print("D2 = Poisson sapma aciklama orani (test, 0=sabit ortalama kadar, 1=mukemmel); sabit_oran = tek carpan;")
    print("model = log-dogrusal Poisson (VIIRS sayi + FRP); model_gunluk = ayrica o gunun toplam VIIRS aktivitesi (rejim).")
    v = cnt["VIIRS_ort"].to_numpy(float)
    f = frp["VIIRS_ort"].to_numpy(float)
    X2 = np.column_stack([np.log1p(v), np.log1p(f)])
    X1 = X2[:, :1]
    daytot = cnt["VIIRS_ort"].groupby(level="gun").transform("sum").to_numpy(float)   # o gunun rejimi
    X3 = np.column_stack([X2, np.log1p(daytot)])
    rows, cals, curve = [], {}, {}
    xs = np.array([0.33, 0.67, 1, 1.33, 2, 3, 5, 8, 12, 20])
    for target in ["Aqua", "Terra"]:
        y = cnt[target].to_numpy(float)
        ytr, yte = y[is_tr], y[~is_tr]
        if ytr.sum() < 20:
            continue
        k = ytr.sum() / max(v[is_tr].sum(), 1e-9)
        p0 = k * v[~is_tr]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m1 = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(X2[is_tr], ytr)
            m3 = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(X3[is_tr], ytr)
            m_all = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(X1, y)
        p1 = m1.predict(X2[~is_tr])
        p3 = m3.predict(X3[~is_tr])
        d_null = _pdev(yte, np.full_like(yte, ytr.mean()))
        rows.append({"hedef": target, "test_hucre_gun": int((~is_tr).sum()),
                     "D2_sabit_oran": 1 - _pdev(yte, p0) / d_null, "D2_model": 1 - _pdev(yte, p1) / d_null,
                     "D2_model_gunluk": 1 - _pdev(yte, p3) / d_null,
                     "Spearman_sabit": pd.Series(p0).corr(pd.Series(yte), method="spearman"),
                     "Spearman_model": pd.Series(p1).corr(pd.Series(yte), method="spearman"),
                     "toplam_gozlenen": yte.sum(), "toplam_sabit": p0.sum(), "toplam_model": p1.sum(),
                     "toplam_model_gunluk": p3.sum()})
        edges = [-0.01, 0.01, 0.5, 1.5, 3.5, 7.5, np.inf]
        names = ["0", "0-0.5", "0.5-1.5", "1.5-3.5", "3.5-7.5", ">7.5"]
        b = pd.cut(pd.Series(v[~is_tr]), edges, labels=names)
        cals[target] = pd.DataFrame({"bin": b, "g": yte, "s": p0, "m": p1}).groupby("bin", observed=True).agg(
            hucre_gun=("g", "size"), gozlenen=("g", "mean"), sabit_oran=("s", "mean"), model=("m", "mean"))
        yhat = m_all.predict(np.log1p(xs)[:, None])
        curve[target] = (yhat, 1 - np.exp(-yhat), float(m_all.coef_[0]))
    if not rows:
        print("  MODIS sayilari cok az, model kurulamadi.")
        return
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    for target, cal in cals.items():
        print(f"\nKALIBRASYON ({target}, test gunleri) - VIIRS aktivitesi (uydu ort. tespit/hucre-gun) araliginda ortalama MODIS sayisi")
        print(cal.round(3).to_string())
    cur = pd.DataFrame({"VIIRS_ort_sayi": xs})
    for target, (yh, p1_, b1) in curve.items():
        cur[f"{target}_beklenen_sayi"] = yh
        cur[f"{target}_P_en_az_1"] = p1_
    print("\nDONUSUM EGRISI (tum gunler, tek ozellikli Poisson; P(>=1) Poisson varsayimiyla)")
    print(cur.round(3).to_string(index=False))
    for target, (yh, p1_, b1) in curve.items():
        yorum = "ust-dogrusal: VIIRS iki katina cikinca MODIS ikiden fazla katina cikar" if b1 > 1.05 else (
            "yaklasik dogrusal" if b1 > 0.95 else "alt-dogrusal")
        print(f"  {target}: beklenen sayi ~ (1+VIIRS)^us, us = {b1:.2f}  -> {yorum}")
    cur.to_csv(OUT_DIR / "transfer_curve.csv", index=False)


def _cellset(df, deg):
    d = pd.to_datetime(df["timestamp"]).dt.normalize()
    return set(zip(np.floor(df["lat"] / deg).astype(int), np.floor(df["lon"] / deg).astype(int), d))


def _cell_agree(A, B, deg, dday, dcell):
    """A'nin hucre-gunlerinin yuzde kacinda B de var (komsuluk toleransiyla)."""
    Bs = _cellset(B, deg)
    offs = [(a, b, pd.Timedelta(days=c)) for a in range(-dcell, dcell + 1)
            for b in range(-dcell, dcell + 1) for c in range(-dday, dday + 1)]
    As = _cellset(A, deg)
    if not As:
        return np.nan, 0
    hit = sum(any((i + a, j + b, d + c) in Bs for a, b, c in offs) for i, j, d in As)
    return 100.0 * hit / len(As), len(As)


def agreement_ladder(modis, viirs):
    """
    Ayni veride MODIS<->VIIRS uyumunu artan olceklerde olcer. Piksel duzeyinde uyum yangin omru ve gecis saati
    farkiyla sinirlidir (literatur: es zamanli gecislerde bile tarim/otlak alanlarinda MODIS->VIIRS ~%47,
    ormanda ~%66; Fu ve ark. 2020). Harmonize takvim icin dogru olcek, uyumun kararli oldugu olcektir.
    """
    rows = []
    half_m = half_size(modis, 0.5)
    half_v = half_size(viirs, 0.1875)
    for hrs, mode in [(6, "ayak izi"), (24, "ayak izi"), (24, "2 km")]:
        nb = neighbors(modis, viirs, 3.0, hrs)
        if mode == "ayak izi":
            hit = [bool(np.any(dd <= half_m[i] + half_v[ii])) for i, (ii, dd, dt) in enumerate(nb)]
        else:
            hit = [bool(np.any(dd <= 2.0)) for (ii, dd, dt) in nb]
        rows.append({"seviye": f"1 piksel ({mode}, +-{hrs} sa)", "MODIS->VIIRS_%": 100 * np.mean(hit),
                     "VIIRS->MODIS_%": np.nan, "n_MODIS": len(modis)})
    for deg, dday, dcell, name in [(0.25, 0, 0, "2 hucre-gun 0.25 derece"), (0.25, 1, 0, "3 hucre-gun 0.25 derece, +-1 gun"),
                                   (0.5, 0, 0, "4 hucre-gun 0.5 derece"), (0.5, 1, 0, "5 hucre-gun 0.5 derece, +-1 gun")]:
        mv, nm = _cell_agree(modis, viirs, deg, dday, dcell)
        vm, nv = _cell_agree(viirs, modis, deg, dday, dcell)
        rows.append({"seviye": name, "MODIS->VIIRS_%": mv, "VIIRS->MODIS_%": vm, "n_MODIS": nm})
    tab = pd.DataFrame(rows)

    A = pd.concat([modis.assign(g="M"), viirs.assign(g="V")], ignore_index=True)
    A["d"] = pd.to_datetime(A["timestamp"]).dt.normalize()
    corr = []
    for deg in (0.5, 1.0):
        i = np.floor(A["lat"] / deg).astype(int)
        j = np.floor(A["lon"] / deg).astype(int)
        tot = A.groupby([i, j, A["g"]]).size().unstack(fill_value=0)
        if {"M", "V"} <= set(tot.columns):
            corr.append({"olcu": f"mekansal desen ({deg:g} derece, tum donem toplami)", "Pearson_log": np.log1p(tot["M"]).corr(np.log1p(tot["V"])),
                         "Spearman": tot["M"].corr(tot["V"], method="spearman"), "n": len(tot)})
        w = A["d"].dt.to_period("W")
        cw = A.groupby([i, j, w, A["g"]]).size().unstack(fill_value=0)
        if {"M", "V"} <= set(cw.columns):
            corr.append({"olcu": f"hucre-hafta ({deg:g} derece)", "Pearson_log": np.log1p(cw["M"]).corr(np.log1p(cw["V"])),
                         "Spearman": cw["M"].corr(cw["V"], method="spearman"), "n": len(cw)})
    dly = A.groupby(["d", "g"]).size().unstack(fill_value=0)
    if {"M", "V"} <= set(dly.columns):
        corr.append({"olcu": "bolge gunluk toplam (zamansal seyir)", "Pearson_log": np.log1p(dly["M"]).corr(np.log1p(dly["V"])),
                     "Spearman": dly["M"].corr(dly["V"], method="spearman"), "n": len(dly)})
    ctab = pd.DataFrame(corr)

    print("\n" + "=" * 66)
    print("UYUM MERDIVENI - ayni veride olcek buyudukce MODIS<->VIIRS uyumu")
    print("=" * 66)
    print(tab.round(1).to_string(index=False))
    print("(VIIRS->MODIS dogal olarak dusuk: VIIRS ~15 kat fazla, cogu MODIS'in gorus esiginin altinda kucuk yangin.)")
    print("\nKORELASYON - desen ve seyir uyumu")
    print(ctab.round(2).to_string(index=False))
    best = tab.dropna(subset=["MODIS->VIIRS_%"])
    top = best[best["MODIS->VIIRS_%"] >= 75]
    if len(top):
        r = top.iloc[0]
        print(f"\n>>> %75 esigini gecen ilk olcek: {r['seviye']}  (MODIS->VIIRS %{r['MODIS->VIIRS_%']:.1f})")
    out = pd.concat([tab.assign(tur="eslesme"), ctab.rename(columns={"olcu": "seviye"}).assign(tur="korelasyon")], ignore_index=True)
    out.to_csv(OUT_DIR / "uyum_merdiveni.csv", index=False)
    return tab, ctab


def persistence_report(modis, viirs):
    """
    VIIRS yangin olaylarini (1 km, 14 saat zincirleme) kac ayri VIIRS gecisinde goruldugune gore siniflar ve
    MODIS'in o olayi [ilk-3 sa, son+3 sa] icinde gorup gormedigini olcer. Kisa omurlu yanginlar MODIS gecisine
    kadar sondugu icin piksel duzeyinde eslesmez; bu tablo bunu dogrudan gosterir.
    """
    lab = cluster_points(viirs, 1.0, 14.0)
    v = viirs.assign(ev=lab, t=pd.to_datetime(viirs["timestamp"]))
    v["pas"] = v["source"].astype(str) + v["t"].dt.floor("90min").astype(str)
    ev = v.groupby("ev").agg(n=("lat", "size"), gecis=("pas", "nunique"), t0=("t", "min"), t1=("t", "max"),
                             frp=("frp", lambda x: pd.to_numeric(x, errors="coerce").sum()))
    ev["sure_sa"] = (ev["t1"] - ev["t0"]).dt.total_seconds() / 3600
    half_m = half_size(modis, 0.5)
    nb = neighbors(v, modis, 3.0, 48)
    mt = pd.to_datetime(modis["timestamp"]).to_numpy()
    t0 = v["ev"].map(ev["t0"]).to_numpy()
    t1 = v["ev"].map(ev["t1"]).to_numpy()
    hit = np.zeros(len(v), bool)
    three = np.timedelta64(3, "h")
    for k, (ii, dd, dt) in enumerate(nb):
        ok = dd <= 0.2 + half_m[ii]
        if ok.any():
            tt = mt[ii[ok]]
            hit[k] = bool(np.any((tt >= t0[k] - three) & (tt <= t1[k] + three)))
    ev["modis"] = pd.Series(hit).groupby(v["ev"].to_numpy()).any()
    ev["kalicilik"] = pd.cut(ev["gecis"], [0, 1, 2, 3, 5, 10_000], labels=["1 gecis", "2 gecis", "3 gecis", "4-5 gecis", ">=6 gecis"])
    g = ev.groupby("kalicilik", observed=True).agg(olay=("n", "size"), MODIS_yakaladi_pct=("modis", lambda x: 100 * x.mean()),
                                                   ort_sure_sa=("sure_sa", "mean"), ort_FRP_MW=("frp", "mean"))
    print("\nYANGIN KALICILIGI - VIIRS olayi kac ayri gecis boyunca goruldu -> MODIS'in yakalama orani")
    print(g.round(1).to_string())
    short = ev["gecis"] <= 2
    print(f"Olaylarin %{100 * short.mean():.0f}'i en fazla 2 gecis (~1 saatten kisa) suruyor; bunlarda MODIS yakalama %{100 * ev.loc[short, 'modis'].mean():.1f}.")
    labm = cluster_points(modis, 2.0, 14.0)
    half_v = half_size(viirs, 0.1875)
    nb2 = neighbors(modis, viirs, 3.0, 14)
    mh = pd.Series([bool(np.any(dd <= half_m[i] + half_v[ii])) for i, (ii, dd, dt) in enumerate(nb2)]).groupby(labm).any()
    sats = modis.assign(ev=labm).groupby("ev")["satellite"].nunique()
    both = sats >= 2
    if both.sum() >= 3:
        print(f"MODIS'in hem Terra (~09:15) hem Aqua (~15:10) ile gordugu kalici olaylar: n={int(both.sum())}, VIIRS de gordu %{100 * mh[both].mean():.0f} "
              f"(tek uydu: %{100 * mh[~both].mean():.0f})")
    g.to_csv(OUT_DIR / "kalicilik.csv")
    return g


# ----------------------------------------------------------------------------
# 5. HARITA
# ----------------------------------------------------------------------------
def _zoom_for(result):
    span = max(result["modis_lat"].max() - result["modis_lat"].min(), result["modis_lon"].max() - result["modis_lon"].min())
    for lim, z in ((25, 5), (12, 6), (6, 7), (3, 8), (1.5, 9)):
        if span > lim:
            return z
    return 10


def make_map(result, path="map.html"):
    import folium
    m = folium.Map(location=[result["modis_lat"].mean(), result["modis_lon"].mean()],
                   zoom_start=_zoom_for(result), tiles=None, prefer_canvas=True)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Harita").add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Uydu").add_to(m)
    g_ok = folium.FeatureGroup(name="Eslesen MODIS (yesil)").add_to(m)
    g_no = folium.FeatureGroup(name="Eslesmeyen MODIS (kirmizi)").add_to(m)

    for _, r in result.iterrows():
        if r["matched"]:
            folium.CircleMarker(
                [r["modis_lat"], r["modis_lon"]], radius=4, color="green",
                fill=True, fill_opacity=0.8,
                popup=(f"MODIS {r['modis_time']}<br>{r['viirs_source']} {r['viirs_time']}"
                       f"<br>{r['distance_km']:.2f} km, {r['time_delta_hours']:.1f} saat"
                       f"<br>penceredeki VIIRS: {r['n_viirs_in_window']}")).add_to(g_ok)
            folium.PolyLine([[r["modis_lat"], r["modis_lon"]],
                             [r["viirs_lat"], r["viirs_lon"]]],
                            color="lime", weight=1, opacity=0.7).add_to(g_ok)
        else:
            folium.CircleMarker(
                [r["modis_lat"], r["modis_lon"]], radius=4, color="red",
                fill=True, fill_opacity=0.8,
                popup=(f"ESLESMEDI<br>{r['modis_time']}<br>guven {r['modis_conf']}"
                       f"<br>FRP {r['modis_frp']}<br>{r['modis_sat']} {r['modis_daynight']}")
            ).add_to(g_no)
    folium.LayerControl().add_to(m)
    m.save(path)
    print(f"\nHarita kaydedildi: {path} (tarayicida ac, sag ustten Uydu'yu sec)")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    key = get_map_key()
    OUT_DIR.mkdir(exist_ok=True)
    start, end = _start_date(), _end_date()
    if end < start:
        sys.exit("Bitis tarihi baslangictan once.")
    print(f"Bolge: {REGION}  kutu (bati,guney,dogu,kuzey): {AREA_BBOX}  |  tarih: {start} .. {end} ({(end - start).days + 1} gun)")
    avail = get_availability(key)
    describe_plan(avail, start, end)
    n_win = max(1, ((end - start).days + 5) // 5)
    if n_win > 6:
        print(f"\n{n_win} pencere x kaynak indirilecek; ilk indirme birkac dakika surebilir.")
    print()

    print("MODIS cekiliyor...")
    modis = standardize(fetch_sensor("MODIS", key, start, end, avail), "MODIS")

    print("VIIRS cekiliyor...")
    frames, failed = [], []
    for sensor in ("VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21"):
        try:
            raw = fetch_sensor(sensor, key, start, end, avail)
            if len(raw):
                frames.append(standardize(raw, sensor))
        except Exception as e:
            failed.append(sensor)
            print(f"  [HATA] {sensor} alinamadi: {e}")
    if failed and not ALLOW_PARTIAL:
        sys.exit("\nSu VIIRS kaynaklari alinamadi: " + ", ".join(failed)
                 + ".\nTekrar calistir: basarili olanlar data_cache/ icinde, sadece eksikler indirilir."
                 + "\n(Eksik veriyle devam etmek icin dosyada ALLOW_PARTIAL = True yap; sonuclar kiyaslanamaz.)")
    viirs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if modis.empty or viirs.empty:
        sys.exit("Bu bolge/tarih icin MODIS ya da VIIRS verisi bos (VIIRS 2012 oncesi yok), eslestirme yapilamiyor.")
    print("VIIRS uydulari:", ", ".join(sorted(viirs["source"].unique())))

    if (datetime.now(timezone.utc).date() - end).days <= 3:      # sadece yakin tarihlerde eksik son gun olabilir
        modis, viirs = trim_incomplete(modis, viirs)
    _tick("kalici kaynak taramasi")
    static_m, static_v, n_days, min_days, nd_m, nd_v = flag_static(modis, viirs)
    report_static(modis, viirs, static_m, static_v, n_days, min_days, nd_m, nd_v)
    if EXCLUDE_STATIC and (static_m.any() or static_v.any()):
        modis = modis[~static_m].reset_index(drop=True)
        viirs = viirs[~static_v].reset_index(drop=True)
        print(f"\n>>> Kalici kaynak adaylari cikarildi. Analiz: MODIS {len(modis)}, VIIRS {len(viirs)}\n")

    _tick("harmonizasyon ve olay kumeleri")
    result, wide, viirs_has_modis = harmonize(modis, viirs)
    labels = cluster_points(viirs, 0.75, 3.0)     # VIIRS: 3 uydunun tekrarlarini birlestir
    labels_m = cluster_points(modis, 2.0, 4.0)    # MODIS: yakin pikseller + Terra/Aqua tekrari
    summarize(result, modis, viirs, viirs_has_modis, labels, labels_m)
    sat_ = modis["satellite"].astype(str)
    has_aqua = reverse_hits(viirs, modis[sat_ == "Aqua"])
    has_terra = reverse_hits(viirs, modis[sat_ == "Terra"])
    _tick("uyum merdiveni ve yangin kaliciligi")
    agreement_ladder(modis, viirs)
    persistence_report(modis, viirs)
    _tick("gunluk veri kalitesi")
    flagged = daily_quality(modis, viirs, result)
    _tick("izgara karsilastirmasi")
    grid_comparison(modis, viirs)
    _tick("VIIRS -> MODIS donusum modeli")
    transfer_model(modis, viirs, skip_days=flagged)
    _tick("verimlilik / FRP / gecis saatleri / matris")
    efficiency_table(viirs, labels, viirs_has_modis, has_aqua, has_terra)
    frp_calibration(modis, viirs, wide)
    pass_time_table(modis, viirs)
    window_table(modis, viirs, wide)
    agreement_matrix(modis, viirs)
    breakdown(result)
    region_report(result)
    sens = sensitivity(wide, show=False)

    result.to_csv(OUT_DIR / "harmonized_output.csv", index=False)
    sens.to_csv(OUT_DIR / "sensitivity.csv")
    print(f"\nKaydedildi ({OUT_DIR.name}/): harmonized_output.csv, sensitivity.csv, grid_cell_day.csv, frp_pairs.csv, transfer_curve.csv, uyum_merdiveni.csv, kalicilik.csv")
    _tick("bitti")

    try:
        make_map(result, str(OUT_DIR / "map.html"))
    except ImportError:
        print("Harita icin: pip install folium")
