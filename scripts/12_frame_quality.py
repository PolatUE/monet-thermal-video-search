"""12: Her video karesi için kalite ölçümleri -> outputs/frame_quality.parquet

Boş kareler (sinyal kopması) MONET'te is_real=True olarak duruyor; seçimde dışlanmaları için ölçülür.
"""
import time

import pandas as pd

from monet.config import OUTPUT_DIR
from monet.data.quality import frame_quality

VIDEO_DIR = OUTPUT_DIR / "videos"
OUT_PATH = OUTPUT_DIR / "frame_quality.parquet"
EMPTY_LIT = 0.01   # yalnızca özet için; tabloya yazılmaz (eşik okuyan kodda)


def main():
    t0 = time.time()
    parts = []
    for path in sorted(VIDEO_DIR.glob("*.mp4")):
        q = frame_quality(path)
        q.insert(0, "clip", path.stem)            # video adı = clip
        parts.append(q)
    quality = pd.concat(parts, ignore_index=True)
    quality.to_parquet(OUT_PATH, index=False)
    print(f"yazildi: {OUT_PATH.name} | {len(quality)} kare | {time.time() - t0:.1f} sn")

    # Kontrol 1: slots tablosundaki her karenin kalite satırı olmalı
    slots = pd.read_parquet(OUTPUT_DIR / "slots.parquet")
    m = slots.merge(quality, on=["clip", "slot"], how="left")
    print("kalite satiri olmayan slot:", int(m["lit"].isna().sum()))

    # Kontrol 2: boş gerçek kareler, klip bazında
    empty = m[m["is_real"] & (m["lit"] < EMPTY_LIT)]
    print(f"bos gercek kare (lit < {EMPTY_LIT}):", len(empty))
    print(empty.groupby("clip")["slot"].agg(["size", "min", "max"]))


if __name__ == "__main__":
    main()