"""
Internetsiz demo icin: harita kutuphanesini (Leaflet) ve yazi tiplerini app/yerel/ altina indirir.
Bir kez, INTERNET VARKEN calistir. Sonra sunucu.py bunlari yerelden verir; internet gitse de harita ve tasarim calisir.

    python cevrimdisi_hazirla.py

Harita karolari (arka plan haritasi) ayri: sadece daha once acilan karolar onbellekte.
Demodan once internet varken, gosterecegin bolgeleri haritada bir kez gezin (yakinlastir/uzaklastir).
"""

import re
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / "app" / "yerel"
LEAF = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/"
LEAF_FILES = ["leaflet.min.js", "leaflet.min.css", "images/layers.png", "images/layers-2x.png",
              "images/marker-icon.png", "images/marker-icon-2x.png", "images/marker-shadow.png"]
FONTS = ("https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;800"
         "&family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&family=Martian+Mono:wght@400;600&display=swap")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/130.0 Safari/537.36"}          # woff2 alabilmek icin modern tarayici gibi gorun


def get(url):
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return r.content


def main():
    n = 0
    for f in LEAF_FILES:
        p = OUT / "leaflet" / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(get(LEAF + f))
        n += 1
    print(f"Leaflet: {n} dosya -> {OUT / 'leaflet'}")

    css = get(FONTS).decode("utf-8")
    urls = sorted(set(re.findall(r"url\((https://fonts\.gstatic\.com/[^)]+)\)", css)))
    (OUT / "fonts").mkdir(parents=True, exist_ok=True)
    for i, u in enumerate(urls):
        name = f"f{i:02d}" + Path(u.split("?")[0]).suffix
        (OUT / "fonts" / name).write_bytes(get(u))
        css = css.replace(u, f"/yerel/fonts/{name}")
    (OUT / "fonts.css").write_text(css, "utf-8")
    print(f"Yazi tipleri: {len(urls)} dosya -> {OUT / 'fonts'}")
    print("Hazir. Sunucuyu yeniden baslat: python sunucu.py")


if __name__ == "__main__":
    main()
