"""13: Segment embedding indeksini üretir -> outputs/index/<ad>/embeddings.parquet + index_info.json

Örnek: python scripts/13_build_index.py
       python scripts/13_build_index.py --frames 8 --name clip_b32_f8   (v2 deneyleri için)
"""
import argparse

import pandas as pd

from monet.config import OUTPUT_DIR
from monet.indexing.build import build_index, save_index
from monet.models.clip import ClipModel


def main():
    p = argparse.ArgumentParser(description="Segment embedding indeksini üret")
    p.add_argument("--frames", type=int, default=4, help="segment başına kare sayısı")
    p.add_argument("--name", default=None, help="indeks klasör adı (varsayılan: model adı)")
    args = p.parse_args()

    seg = pd.read_parquet(OUTPUT_DIR / "segments.parquet")
    slots = pd.read_parquet(OUTPUT_DIR / "slots.parquet")
    quality = pd.read_parquet(OUTPUT_DIR / "frame_quality.parquet")   # 12 numaralı adımın çıktısı
    model = ClipModel()

    table, vectors, info = build_index(model, seg, slots, quality=quality, n=args.frames)
    out_dir = save_index(table, vectors, info, args.name or model.name)

    print(f"yazildi: {out_dir}")
    print(f"segment: {info['segments']} | kare: {info['frames']} | sure: {info['build_seconds']} sn")
    print("n_frames dagilimi:", table["n_frames"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()