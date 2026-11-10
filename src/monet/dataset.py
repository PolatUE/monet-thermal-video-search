import json
import math
import re
from pathlib import Path

import pandas as pd

from monet.config import RAW_DIR

# Her sekansta etiket dosyasının göreli konumu
LABELS_REL = Path("annotations") / "labels.json"

# metadata içindeki 12 telemetri alanı; sütun sırası bu listeye göre
TELEMETRY_KEYS = [
    "UAV_roll", "UAV_pitch", "UAV_yaw",
    "gimbal_roll", "gimbal_pitch", "gimbal_yaw",
    "latitude", "longitude", "altitude",
    "speed_x", "speed_y", "speed_z",
]

# "image2501.png" -> 2501 ; sondaki sayıyı alır
_FRAME_RE = re.compile(r"(\d+)\.png$")


def find_sequences(raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    """İçinde annotations/labels.json olan klasörleri sekans sayar (ör. .venv elenir)."""
    seqs = {}
    for d in sorted(raw_dir.iterdir()):
        if d.is_dir() and (d / LABELS_REL).is_file():
            seqs[d.name] = d
    return seqs


def load_labels(seq_dir: Path) -> dict:
    """labels.json'u yalnızca okuma modunda açar; ham veriye hiçbir şey yazılmaz."""
    with open(seq_dir / LABELS_REL, "r", encoding="utf-8") as f:
        return json.load(f)


def to_float(v) -> float:
    """Sayıya çevrilebiliyorsa sayı, değilse ('' veya None) NaN döndürür."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def frame_number(file_name: str) -> int:
    """Dosya adından orijinal kare numarasını çıkarır; beklenmedik ad olursa hata verir."""
    m = _FRAME_RE.search(file_name)
    if m is None:
        raise ValueError(f"Kare numarası okunamadı: {file_name}")
    return int(m.group(1))


def frames_table(seq_name: str, labels: dict) -> pd.DataFrame:
    """Bir sekansın her karesi için bir satır: kimlik, kare no, zaman ve 12 telemetri sütunu."""
    rows = []
    for im in labels["images"]:
        meta = im.get("metadata") or {}  # metadata hiç yoksa boş sözlük say
        row = {
            "seq": seq_name,
            "image_id": im["id"],          # kutuları (annotations) karelere bağlamak için
            "file_name": im["file_name"],
            "frame_no": frame_number(im["file_name"]),
            "date_captured": im["date_captured"],
        }
        for k in TELEMETRY_KEYS:
            row[k] = to_float(meta.get(k))
        rows.append(row)

    df = pd.DataFrame(rows)
    # Zamanı UTC'ye çevir: farklı saat dilimi gelse bile tek eksende karşılaştırılır
    df["t"] = pd.to_datetime(df["date_captured"], utc=True)
    df = df.drop(columns="date_captured")
    return df.sort_values("frame_no").reset_index(drop=True)

def annotations_table(seq_name: str, labels: dict) -> pd.DataFrame:
    """Her kutu için bir satır: hangi kare, sınıf, konum/boyut, iz numarası."""
    cats = {c["id"]: c["name"] for c in labels["categories"]}
    rows = []
    for a in labels["annotations"]:
        x, y, w, h = a["bbox"]
        attr = a.get("attributes") or {}
        rows.append({
            "seq": seq_name, "ann_id": a["id"], "image_id": a["image_id"],
            "category": cats[a["category_id"]],
            "x": x, "y": y, "w": w, "h": h,
            "track_id": attr.get("track_id"), "occluded": attr.get("occluded"),
        })
    return pd.DataFrame(rows)