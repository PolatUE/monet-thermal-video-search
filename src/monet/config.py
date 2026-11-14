import os
from pathlib import Path

# Proje kökü bu dosyanın konumundan hesaplanır: src/monet/config.py -> 2 klasör yukarı
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Ham veri: YALNIZCA OKUNUR. Yol koda yazılmaz; makineye özgü MONET_RAW_DIR ortam
# değişkeninden okunur. Tanımlı değilse proje içindeki data/monet_dataset varsayılır.
RAW_DIR = Path(os.environ.get("MONET_RAW_DIR", PROJECT_DIR / "data" / "monet_dataset"))

OUTPUT_DIR = PROJECT_DIR / "outputs"
REPORT_DIR = PROJECT_DIR / "reports"

# Güvenlik: çıktı klasörleri ham verinin içinde olamaz (ham veriye yazmayı önler)
_raw = RAW_DIR.resolve()
for _d in (OUTPUT_DIR, REPORT_DIR):
    if _d.resolve().is_relative_to(_raw):
        raise RuntimeError(f"Çıktı klasörü ham veri altında olamaz: {_d}")

INDEX_DIR = OUTPUT_DIR / "index"   # model başına embedding indeksleri: index/<model_adı>/