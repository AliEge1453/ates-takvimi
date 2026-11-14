"""
Hazir veri seti bolgeleri (oncelik sirasiyla: en yakin tarihli buyuk yanginlar once).
toplu_indir.py bu listeyi sirayla indirir; sunucu.py bu bolgelerin icindeki her secimi aninda hesaplar.
kutu: "bati,guney,dogu,kuzey" (derece). yillar: varsayilan 2012-2026 (VIIRS donemi; 2012-2017 kalibrasyon, 2018-2020 test).
"""

BOLGELER = [
    # --- 2026 ---
    {"key": "ES_MERKEZ", "ad": "İspanya: Madrid, Ávila, Toledo, Guadalajara", "kutu": "-5.5,39.5,-2.0,41.5",
     "olay": "2026: Burgohondo–Villa del Prado (~50.000 ha, İspanya'nın kayıtlı en büyük yangını), La Mierla"},
    {"key": "ES_ENDULUS", "ad": "İspanya: Huelva ve Sevilla", "kutu": "-7.6,36.8,-5.0,38.3",
     "olay": "2026: Niebla yangını (38.000+ ha, Endülüs'ün en kötüsü)"},
    {"key": "FR_GIRONDE", "ad": "Fransa: Gironde ve Landes", "kutu": "-1.6,43.6,0.3,45.6",
     "olay": "2026: Gironde/Landes (~45.000 ha, 1949'dan beri Fransa'nın en büyüğü); 2022: Landiras"},
    {"key": "IBERYA_KB", "ad": "Kuzey Portekiz ve Galiçya", "kutu": "-9.3,40.4,-6.2,43.8",
     "olay": "2026: Valpaços, Vouzela; 2025: Ourense; 2017: Pedrógão Grande"},
    {"key": "GR_ATTIKA", "ad": "Yunanistan: Attika, Boiotia, Evia", "kutu": "22.4,37.6,24.6,39.2",
     "olay": "2026: Porto Germeno; 2024: Attika; 2021: Evia"},
    {"key": "US_OREGON", "ad": "ABD: Oregon", "kutu": "-124.6,41.9,-116.4,46.3",
     "olay": "2026: eyalet rekoruna yaklaşan sezon (2 milyon+ akre; Big Grass, Rowe Creek, Coleman Creek)"},
    {"key": "GR_GIRIT", "ad": "Yunanistan: Girit", "kutu": "23.4,34.8,26.4,35.8",
     "olay": "2026: Rethymno yangınları"},
    # --- 2025 ---
    {"key": "LEVANT", "ad": "İsrail, Lübnan, Batı Suriye", "kutu": "34.2,31.0,36.9,35.9",
     "olay": "2025: Lazkiye (Temmuz), Kudüs tepeleri (Nisan)"},
    {"key": "KIBRIS", "ad": "Kıbrıs", "kutu": "32.2,34.5,34.7,35.8",
     "olay": "2025: Limasol yangını"},
    {"key": "KORE", "ad": "Güney Kore", "kutu": "125.8,34.0,129.7,38.7",
     "olay": "2025 Mart: Uiseong–Andong, ülke tarihinin en büyük yangınları"},
    {"key": "BALKAN", "ad": "Arnavutluk, Kuzey Makedonya, Karadağ", "kutu": "18.4,39.6,23.0,43.6",
     "olay": "2025 yazı: Balkanlar'da yaygın yangınlar"},
    # --- 2024 ---
    {"key": "SILI", "ad": "Orta Şili: Valparaíso–Biobío", "kutu": "-73.8,-38.5,-70.5,-32.5",
     "olay": "2024 Şubat: Viña del Mar; 2023: Ñuble–Biobío"},
    {"key": "PANTANAL", "ad": "Pantanal, Brezilya", "kutu": "-59.0,-22.0,-54.5,-15.0",
     "olay": "2024 ve 2020: rekor Pantanal yangınları"},
    {"key": "KANADA_JASPER", "ad": "Kanada: Jasper", "kutu": "-119.5,52.0,-116.5,53.8",
     "olay": "2024 Temmuz: Jasper kasabası yangını"},
    # --- daha eski, iyi bilinen ---
    {"key": "HAVAI_MAUI", "ad": "Havai: Maui", "kutu": "-156.8,20.5,-155.9,21.1",
     "olay": "2023 Ağustos: Lahaina"},
]

# Mevcut uc bolge (zaten indirildi)
MEVCUT = [
    {"key": "TR", "ad": "Türkiye ve çevresi", "kutu": "25,35,45,43", "olay": "2021: Manavgat, Marmaris; 2025: İzmir; 2003–2026 tam seri"},
    {"key": "LA", "ad": "Los Angeles", "kutu": "-119.0,33.6,-117.4,34.6", "olay": "2025 Ocak: Palisades ve Eaton"},
    {"key": "NSW", "ad": "Yeni Güney Galler, Avustralya", "kutu": "140.9,-37.6,153.7,-28.0", "olay": "2019–20: Kara Yaz"},
]

HEPSI = {b["key"]: b for b in MEVCUT + BOLGELER}
