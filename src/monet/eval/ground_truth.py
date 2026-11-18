"""Değerlendirme için doğru cevaplar: kutuları segmentlere bağlar, her segmenti sorguya göre işaretler."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from monet.config import OUTPUT_DIR, PROJECT_DIR
from monet.indexing.frames import FPS

QUERIES_PATH = PROJECT_DIR / "eval" / "queries.json"
CLASSES = ["person", "vehicle"]


def load_queries(path: Path = QUERIES_PATH) -> dict:
    """Sorgu dosyasını okur: sürüm + sorgu türleri (id, target, texts)."""
    return json.loads(path.read_text(encoding="utf-8"))


def segment_labels(min_lit: float = 0.01) -> pd.DataFrame:
    """Segment başına: sınıf var mı (has_*), en büyük kutu (*_px, ortalama kenar) ve kamera açısı.

    Yalnızca gerçek (is_real) ve boş olmayan (lit >= min_lit) karelerdeki kutular sayılır:
    tekrar karelerde aynı kutu iki kez sayılmasın, boş karelerdeki görünmeyen kutular doğru cevap olmasın.
    """
    seg = pd.read_parquet(OUTPUT_DIR / "segments.parquet")
    slots = pd.read_parquet(OUTPUT_DIR / "slots.parquet")
    quality = pd.read_parquet(OUTPUT_DIR / "frame_quality.parquet")
    boxes = pd.read_parquet(OUTPUT_DIR / "suitability" / "box_stats.parquet")

    # Gerçek ve dolu video kareleri, sekans adıyla (kutular sekans + dosya adıyla eşleşir)
    real = (slots[slots["is_real"]]
            .merge(seg[["clip", "seq"]].drop_duplicates(), on="clip")
            .merge(quality[["clip", "slot", "lit"]], on=["clip", "slot"]))
    real = real[real["lit"] >= min_lit]
    x = boxes.merge(real[["seq", "src_file", "clip", "slot"]],
                    left_on=["seq", "file_name"], right_on=["seq", "src_file"])

    # Video karesi -> segment: kendinden önceki en yakın segment başlangıcı, uzunluk (yuva) içindeyse
    starts = seg[["clip", "seg", "video_bas", "yuva"]].assign(
        start=(seg["video_bas"] * FPS).round().astype(int))
    x = pd.merge_asof(x.sort_values("slot"), starts.sort_values("start"),
                      left_on="slot", right_on="start", by="clip", direction="backward")
    x = x[x["slot"] < x["start"] + x["yuva"]]

    # Sınıf başına segmentteki en büyük kutu; kutusu olmayan segmentte boş (NaN)
    size = (x.pivot_table(index=["clip", "seg"], columns="category", values="size", aggfunc="max")
             .reindex(columns=CLASSES))
    size.columns = [f"{c}_px" for c in CLASSES]
    out = seg[["clip", "seg", "seq", "gimbal_pitch"]].merge(size.reset_index(), on=["clip", "seg"], how="left")
    for c in CLASSES:
        out[f"has_{c}"] = out[f"{c}_px"].notna()
    return out


def relevance(labels: pd.DataFrame, target: dict) -> pd.DataFrame:
    """Her segment için rel: 1 doğru, 0 yanlış, NaN değerlendirme dışı; group: nesnede 'big' / 'small'."""
    if target["type"] == "object":
        c = target["category"]
        has = labels[f"has_{c}"]
        rel = has.astype(float)
        big = labels[f"{c}_px"] >= target["big_px"]
        group = np.where(~has, None, np.where(big, "big", "small"))
    elif target["type"] == "pitch":
        p = labels["gimbal_pitch"]
        rel = pd.Series(np.nan, index=labels.index)          # telemetrisiz ve sınır bölgesi: dışarıda
        rel[p <= target["pos_max_deg"]] = 1.0
        rel[p >= target["neg_min_deg"]] = 0.0
        group = np.full(len(labels), None)
    else:
        raise ValueError(f"bilinmeyen hedef tipi: {target['type']}")
    return pd.DataFrame({"rel": rel, "group": group}, index=labels.index)