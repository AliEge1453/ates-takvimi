"""
Ates Takvimi'ni baska bir bilgisayara tasimak icin tek zip hazirlar (hicbir veri yeniden indirilmez).

    cd C:\\nasa_harmonization
    python paketle.py                 ->  paket\\AtesTakvimi_<tarih>.zip

Icine girenler: tum .py dosyalari, app\\, dataset\\ (dunya verisi dahil), cikti_TR\\, belgeler, harita ve yer arama onbellegi.
Girmeyenler: data_cache icindeki ham FIRMS dosyalari (cok buyuk, gerekmez) ve map_key.txt (kisisel anahtar).
Yeni bilgisayarda: zip'i ac, Python kur, `pip install -r requirements.txt`, sonra `python sunucu.py`.
Hazir bolgeler ve dunya verisi anahtarsiz calisir; canli indirme icin map_key.txt'yi elle kopyala.
"""

import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "paket"
SKIP_DIRS = {"__pycache__", "paket", ".git"}
CACHE_KEEP = {"geo", "karolar"}             # data_cache icinden sadece bunlar (kucuk, internetsiz demo icin faydali)


def files():
    for p in sorted(HERE.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(HERE)
        parts = rel.parts
        if parts[0] in SKIP_DIRS or any(x in SKIP_DIRS for x in parts):
            continue
        if parts[0] == "data_cache" and (len(parts) < 3 or parts[1] not in CACHE_KEEP):
            continue
        if rel.name == "map_key.txt" and "--anahtar" not in sys.argv:
            continue
        if rel.suffix in (".log", ".tmp"):
            continue
        yield p, rel


def main():
    OUT.mkdir(exist_ok=True)
    name = OUT / f"AtesTakvimi_{time.strftime('%Y%m%d_%H%M')}.zip"
    n, raw = 0, 0
    with zipfile.ZipFile(name, "w") as z:
        for p, rel in files():
            comp = zipfile.ZIP_STORED if p.suffix in (".gz", ".zip", ".png", ".jpg") else zipfile.ZIP_DEFLATED
            z.write(p, Path("nasa_harmonization") / rel, compress_type=comp)
            n += 1
            raw += p.stat().st_size
    print(f"Hazir: {name}")
    print(f"  {n} dosya, {raw / 1e6:,.0f} MB veri -> zip {name.stat().st_size / 1e6:,.0f} MB")
    print("  map_key.txt pakete konmadi (kisisel). Canli indirme gerekirse elle kopyala ya da --anahtar ile ekle.")
    print("  Yeni bilgisayarda: zip'i ac, 'pip install -r requirements.txt', sonra 'python sunucu.py'.")


if __name__ == "__main__":
    main()
