"""Segment embedding indeksini üretir, kaydeder ve yükler: outputs/index/<ad>/"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from monet.config import INDEX_DIR, OUTPUT_DIR
from monet.indexing.frames import pad_to_square, read_frames, select_frames


def build_index(model, segments: pd.DataFrame, slots: pd.DataFrame, quality: pd.DataFrame | None = None,
                video_dir: Path = OUTPUT_DIR / "videos", n: int = 4, real_only: bool = True,
                min_lit: float = 0.01) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Her segment için bir vektör üretir.

    Dönüş: (tablo: clip, seg, n_frames), (vektörler: segment sayısı x boyut, birim uzunluk), (ayarlar sözlüğü).
    Tablo ve vektörler aynı sırada: tablonun i. satırı = vektörlerin i. satırı.
    """
    t0 = time.time()
    sel = select_frames(segments, slots, n=n, real_only=real_only, quality=quality, min_lit=min_lit)

    # Kareleri klip klip oku ve kodla: her video bir kez çözülür
    keys, vecs = [], []
    for clip, s in sel.groupby("clip", sort=False):
        imgs = dict(read_frames(video_dir / f"{clip}.mp4", s["slot"]))
        vecs.append(model.encode_images([pad_to_square(imgs[x]) for x in s["slot"]]))
        keys += list(zip(s["clip"], s["seg"]))
    frame_vecs = np.concatenate(vecs)

    # Segment vektörü = kare vektörlerinin ortalaması, sonra yeniden birim uzunluk
    fr = pd.DataFrame(keys, columns=["clip", "seg"]).assign(i=np.arange(len(keys)))
    groups = fr.groupby(["clip", "seg"], sort=True)["i"].apply(np.array)
    seg_vecs = np.stack([frame_vecs[ix].mean(axis=0) for ix in groups])
    seg_vecs /= np.linalg.norm(seg_vecs, axis=1, keepdims=True)

    table = groups.index.to_frame(index=False)
    table["n_frames"] = [len(ix) for ix in groups]       # 4'ten azsa segment az kareyle temsil edildi

    info = {
        "model": model.name, "arch": model.arch, "pretrained": model.pretrained, "dim": int(model.dim),
        "frames_per_segment": n, "real_only": real_only,
        "min_lit": min_lit if quality is not None else None,
        "preprocess": "pad_to_square(black) + model preprocess",
        "segments": len(table), "frames": len(frame_vecs),
        "build_seconds": round(time.time() - t0, 1),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return table, seg_vecs.astype(np.float32), info


def save_index(table: pd.DataFrame, vectors: np.ndarray, info: dict, name: str) -> Path:
    """index/<name>/ altına embeddings.parquet (anahtar + vektör) ve index_info.json (ayarlar) yazar."""
    out_dir = INDEX_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    df = table.copy()
    df["emb"] = list(vectors)                            # her satıra kendi vektörü (Parquet'te liste sütunu)
    df.to_parquet(out_dir / "embeddings.parquet", index=False)
    (out_dir / "index_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def load_index(name: str) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """save_index'in tersi: (tablo, vektör matrisi, ayarlar)."""
    in_dir = INDEX_DIR / name
    df = pd.read_parquet(in_dir / "embeddings.parquet")
    vectors = np.stack(df.pop("emb").to_numpy()).astype(np.float32)   # liste sütunu -> (N, D) matris
    info = json.loads((in_dir / "index_info.json").read_text(encoding="utf-8"))
    return df, vectors, info