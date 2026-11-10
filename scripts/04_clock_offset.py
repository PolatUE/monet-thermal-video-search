import sys

import matplotlib
matplotlib.use("Agg")  # pencere açmadan doğrudan dosyaya çiz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet.config import OUTPUT_DIR

REF_FPS = 25.0  # karşılaştırma için sabit kamera hızı
# Varsayılan: dirtroad_04 sorunsuz referans, diğer ikisi sorunlu çıkanlar
SEQS = sys.argv[1:] or ["dirtroad_04", "dirtroad_05", "dirtroad_07"]


def clock_offset(df: pd.DataFrame):
    """Her kare için: kayıttaki zaman - kare numarasından 25 fps ile beklenen zaman (sn)."""
    df = df.sort_values("frame_no")
    n = df["frame_no"].to_numpy()
    ts = (df["t"] - df["t"].iloc[0]).dt.total_seconds().to_numpy()
    vt = (n - n[0]) / REF_FPS  # video zamanı: yalnızca kare numarasından
    return n, vt, ts - vt


frames = pd.read_parquet(OUTPUT_DIR / "frames.parquet")
out_dir = OUTPUT_DIR / "suitability"
out_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(len(SEQS), 1, figsize=(10, 2.6 * len(SEQS)),
                         sharex=True, sharey=True)  # ortak eksen: dürüst karşılaştırma
axes = np.atleast_1d(axes)

for ax, seq in zip(axes, SEQS):
    n, vt, off = clock_offset(frames[frames["seq"] == seq])

    # Ardışık iki kare arasında saatin beklenenden en çok ilerlediği 3 yer
    extra = np.diff(off)
    print(f"\n{seq}: sondaki toplam fark {off[-1]:+.1f} sn")
    for i in np.argsort(extra)[::-1][:3]:
        print(f"  image{n[i]} -> image{n[i + 1]}  video {vt[i]:6.1f}. sn  "
              f"saat {extra[i]:+.1f} sn fazla ilerlemiş")

    ax.plot(vt, off, color="#2a78d6", linewidth=1.5)
    for y in (-2, -1, 1, 2):  # ±1 sn yuvarlama payı, ±2 sn sorun eşiği
        ax.axhline(y, color="#b0aea8", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set_title(seq, loc="left", fontsize=11)
    ax.set_ylabel("fark (sn)")
    ax.spines[["top", "right"]].set_visible(False)

axes[-1].set_xlabel("video zamanı (sn) = (kare no - ilk kare no) / 25")
fig.suptitle("Kayıttaki zaman - kare numarasından beklenen zaman", x=0.01, ha="left")
fig.tight_layout()
out_path = out_dir / "clock_offset.png"
fig.savefig(out_path, dpi=150)
print(f"\nKaydedildi: {out_path}")