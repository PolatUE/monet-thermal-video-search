import numpy as np
from PIL import Image

from monet.data.dataset import find_sequences, frame_number

for name, d in find_sequences().items():
    # Kare numarasına göre sırala (alfabetik sırada image10 < image2 olur)
    files = sorted((d / "lwir").glob("*.png"), key=lambda p: frame_number(p.name))
    for p in (files[0], files[len(files) // 2], files[-1]):  # ilk, orta, son
        with Image.open(p) as im:  # with: dosya okunup hemen kapanır
            mode, size = im.mode, im.size
            a = np.asarray(im)
        ch = a.shape[2] if a.ndim == 3 else 1
        print(f"{name:12s} {p.name:16s} mode={mode:6s} boyut={size} kanal={ch} "
              f"dtype={a.dtype} min={a.min()} max={a.max()}")