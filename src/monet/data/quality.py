"""Kare kalite ölçümleri: boş/sinyalsiz kareleri ayırt etmek için parlaklık ve aydınlık piksel oranı."""
from pathlib import Path

import av
import pandas as pd

LIT_LEVEL = 10    # 0–255 gri ölçekte bundan parlak piksel "aydınlık" sayılır


def frame_quality(video_path: Path) -> pd.DataFrame:
    """Videoyu bir kez çözer; her kare için slot, ortalama parlaklık (mean) ve aydınlık piksel oranı (lit).

    lit, birkaç parlak piksele duyarsız olduğu için boş kareyi mean ya da max'tan daha güvenilir ayırır.
    Karar (eşik) burada verilmez; tabloyu okuyan kod eşiği uygular.
    """
    rows = []
    with av.open(str(video_path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"                      # çözmeyi çekirdeklere yay
        fps = float(s.average_rate)                 # slot = zaman x kare hızı (videonun kendisinden)
        for f in c.decode(s):
            g = f.to_ndarray(format="gray")         # yalnızca parlaklık kanalı; renk gerekmiyor
            rows.append((round(f.time * fps), float(g.mean()), float((g > LIT_LEVEL).mean())))
    return pd.DataFrame(rows, columns=["slot", "mean", "lit"])