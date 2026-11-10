import sys
import time
from fractions import Fraction

import av
import numpy as np
import pandas as pd
from PIL import Image

from monet.config import OUTPUT_DIR, RAW_DIR

# Kodlama ayarları (tek yerde)
VIDEO_FPS = Fraction(25, 2)  # 12,5 fps; kesir olarak yazılır ki yuvarlama hatası olmasın
CRF = 18                     # kalite: düşük = daha iyi kalite, daha büyük dosya
PRESET = "veryfast"          # kodlama hızı / dosya boyutu dengesi
VIDEO_DIR = OUTPUT_DIR / "videos"


def encode_clip(clip: str, seq: str, s: pd.DataFrame) -> None:
    out = VIDEO_DIR / f"{clip}.mp4"
    tmp = VIDEO_DIR / f"{clip}.part.mp4"  # yarım kalan dosya bitmiş sanılmasın
    lwir = RAW_DIR / seq / "lwir"         # ham veri yalnızca okunur

    t0 = time.perf_counter()
    with av.open(str(tmp), mode="w") as container:
        stream = container.add_stream("libx264", rate=VIDEO_FPS)
        stream.width, stream.height = 800, 600
        stream.pix_fmt = "yuv420p"  # en yaygın oynatılabilir piksel formatı
        stream.options = {"crf": str(CRF), "preset": PRESET}

        last_file, rgb = None, None
        for i, f in enumerate(s["src_file"]):
            if f != last_file:  # tekrar karede PNG'yi yeniden okumaya gerek yok
                with Image.open(lwir / f) as im:
                    rgb = np.asarray(im.convert("RGB"))
                last_file = f
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = i  # i. yuva = i / 12,5 sn; zamanı yuva numarası belirler
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():  # kodlayıcıda bekleyen kareleri boşalt
            container.mux(packet)
    tmp.replace(out)

    # Doğrulama: dosyadaki kare sayısı ve hız plana uyuyor mu?
    with av.open(str(out)) as c:
        vs = c.streams.video[0]
        n_file, rate = vs.frames, vs.average_rate
    ok = "OK" if (n_file == len(s) and rate == VIDEO_FPS) else "UYUMSUZ"
    print(f"{clip:16s} {len(s):5d} kare  {len(s) / float(VIDEO_FPS):7.2f} sn  "
          f"%{100 * s['is_real'].mean():5.1f} gerçek  {out.stat().st_size / 1e6:6.1f} MB  "
          f"{time.perf_counter() - t0:5.1f} sn  dosya: {n_file} kare @ {float(rate)} fps  {ok}")


clips = pd.read_parquet(OUTPUT_DIR / "clips.parquet")
slots = pd.read_parquet(OUTPUT_DIR / "slots.parquet")
todo = clips[clips["durum"] == "video"]
if len(sys.argv) > 1:  # isteğe bağlı: yalnızca adı verilen klipler (deneme için)
    todo = todo[todo["clip"].isin(sys.argv[1:])]

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
for _, c in todo.iterrows():
    if (VIDEO_DIR / f"{c['clip']}.mp4").exists():  # yarıda kesilirse kaldığı yerden devam
        print(f"{c['clip']:16s} zaten var, atlandı")
        continue
    s = slots[slots["clip"] == c["clip"]].sort_values("slot")
    encode_clip(c["clip"], c["seq"], s) 