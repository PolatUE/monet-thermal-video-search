import numpy as np
import pandas as pd

from monet.config import OUTPUT_DIR

# Onaylanan kurallar (tek yerde)
SRC_FPS = 25.0          # kamera hızı (3a)
VIDEO_FPS = 12.5        # üretilecek video hızı
SPLIT_GAP_SEC = 5.0     # bundan uzun boşlukta sekans bölünür
MIN_CLIP_SEC = 5.0      # bundan kısa parça videoya çevrilmez

STEP = round(SRC_FPS / VIDEO_FPS)  # bir video karesi = 2 kamera karesi
assert STEP * VIDEO_FPS == SRC_FPS, "Video fps kamera fps'ini tam bölmeli"


def plan_sequence(seq: str, df: pd.DataFrame):
    df = df.sort_values("frame_no").reset_index(drop=True)
    n = df["frame_no"].to_numpy()

    # Boşluk > 5 sn olan yerde yeni parça başlar; ilk kare her zaman parça başı
    gap_sec = np.diff(n, prepend=n[0]) / SRC_FPS
    df["part"] = np.cumsum(gap_sec > SPLIT_GAP_SEC) + 1

    clips, slots = [], []
    for part, real in df.groupby("part"):
        name = f"{seq}_p{part}"
        first, last = int(real["frame_no"].iloc[0]), int(real["frame_no"].iloc[-1])

        # Izgara kontrolü: her gerçek kare bir yuvaya tam oturmalı (hepsi tek sayı -> fark çift)
        if ((real["frame_no"] - first) % STEP != 0).any():
            raise ValueError(f"{name}: kare numaraları 12,5 fps ızgarasına oturmuyor")

        n_slots = (last - first) // STEP + 1
        dur = n_slots / VIDEO_FPS
        keep = dur >= MIN_CLIP_SEC

        clips.append({
            "clip": name, "seq": seq,
            "durum": "video" if keep else "atlandı (< 5 sn)",
            "ilk_kare": first, "son_kare": last,
            "t_bas": real["t"].iloc[0], "t_son": real["t"].iloc[-1],
            "sure_sn": round(dur, 2), "yuva": n_slots,
            "gercek": len(real), "gercek_%": round(100 * len(real) / n_slots, 1),
        })
        if not keep:
            continue

        # Her yuva için kamera kare numarası; kaynağı = o numaraya kadar gelen son gerçek kare
        s = pd.DataFrame({"slot": np.arange(n_slots)})
        s["frame_no"] = first + STEP * s["slot"]
        src = real[["frame_no", "file_name", "t"]].rename(
            columns={"frame_no": "src_frame_no", "file_name": "src_file", "t": "t_src"})
        s = pd.merge_asof(s, src, left_on="frame_no", right_on="src_frame_no",
                          direction="backward")  # boşlukta önceki kare tekrar eder
        s["is_real"] = s["frame_no"] == s["src_frame_no"]
        s["video_sec"] = s["slot"] / VIDEO_FPS
        s.insert(0, "clip", name)
        slots.append(s)

    return clips, slots


frames = pd.read_parquet(OUTPUT_DIR / "frames.parquet")
all_clips, all_slots = [], []
for seq, g in frames.groupby("seq"):
    c, s = plan_sequence(seq, g)
    all_clips += c
    all_slots += s

clips = pd.DataFrame(all_clips)
slots = pd.concat(all_slots, ignore_index=True)
clips.to_parquet(OUTPUT_DIR / "clips.parquet", index=False)
slots.to_parquet(OUTPUT_DIR / "slots.parquet", index=False)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print(clips.drop(columns=["t_bas", "t_son"]).to_string(index=False))
v = clips[clips["durum"] == "video"]
print(f"\nVideo: {len(v)} klip, {v['sure_sn'].sum() / 60:.1f} dk, "
      f"{len(slots)} video karesi ({slots['is_real'].mean() * 100:.1f}% gerçek)")
print(f"Atlanan: {(clips['durum'] != 'video').sum()} parça")