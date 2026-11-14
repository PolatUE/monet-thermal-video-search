"""Segment başına embedding'e girecek kareleri seçer (videodan okuma ve kenar ekleme sonraki adımda)."""
from pathlib import Path
from typing import Iterable, Iterator

import av
from PIL import Image
import numpy as np
import pandas as pd

FPS = 12.5        # videolar sabit 12,5 fps; slot numarası = video kare numarası (kontrol edildi)
# Segment sınırları segments.parquet'ten gelir: başlangıç = video_bas * FPS, uzunluk = yuva (62 ya da 63)

def select_frames(segments: pd.DataFrame, slots: pd.DataFrame, n: int = 4,
                  real_only: bool = True, quality: pd.DataFrame | None = None,
                  min_lit: float = 0.01) -> pd.DataFrame:
    """Her segmentten zamana eşit yayılmış en fazla n kare seçer.

    Kural zamana dayalı, bu yüzden gerçek videolarda da aynen çalışır. Adayları iki filtre daraltır:
    - real_only: yalnızca MONET'e özgü; tekrar kareleri (is_real=False) aday olmaz.
    - quality + min_lit: genel; aydınlık piksel oranı min_lit'in altındaki boş kareler aday olmaz.
    Dönüş: her satır bir seçilmiş kare -> clip, seg, slot, video_sec.
    """
    cand = slots[slots["is_real"]] if real_only else slots
    if quality is not None:
        # Kalite tablosuyla eşleştir, boş kareleri at (eşik burada, tabloda değil)
        cand = cand.merge(quality[["clip", "slot", "lit"]], on=["clip", "slot"], how="left")
        cand = cand[cand["lit"] >= min_lit]
    # Klip başına sıralı slot dizisi: her segmentte tüm tabloyu taramamak için
    by_clip = {k: g["slot"].sort_values().to_numpy() for k, g in cand.groupby("clip")}

    rows = []
    for clip, seg, bas, length in segments[["clip", "seg", "video_bas", "yuva"]].itertuples(index=False):
        start = int(round(bas * FPS))                          # segmentin ilk slotu
        s = by_clip.get(clip, np.empty(0, dtype=np.int64))
        s = s[(s >= start) & (s < start + length)]             # bu segmentin aday kareleri
        if len(s) == 0:
            continue
        # Hedefler: segmenti n eşit parçaya böl, her parçanın ortası
        targets = start + (np.arange(n) + 0.5) * length / n
        # Her hedefe en yakın aday; aynı kare iki hedefe yakınsa bir kez alınır
        picked = np.unique(s[np.abs(s[:, None] - targets[None, :]).argmin(axis=0)])
        rows += [(clip, seg, int(p)) for p in picked]

    out = pd.DataFrame(rows, columns=["clip", "seg", "slot"])
    out["video_sec"] = out["slot"] / FPS
    return out

def read_frames(video_path: Path, wanted: Iterable[int]) -> Iterator[tuple[int, Image.Image]]:
    """Videoyu bir kez sırayla çözer; yalnızca istenen slotları (slot, PIL görüntü) olarak döndürür."""
    wanted = {int(w) for w in wanted}
    if not wanted:
        return
    last = max(wanted)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"              # çözmeyi birden çok çekirdeğe yay
        for frame in container.decode(stream):
            idx = round(frame.time * FPS)        # sunum zamanı -> slot (sayaca güvenmeden)
            if idx in wanted:
                yield idx, frame.to_image()
            if idx >= last:
                break                            # son istenen kareden sonrasını çözme


def pad_to_square(img: Image.Image, fill=(0, 0, 0)) -> Image.Image:
    """Kareyi kırpmadan kare biçime getirir: kısa kenara eşit dolgu ekler (800x600 -> 800x800)."""
    w, h = img.size
    side = max(w, h)
    out = Image.new("RGB", (side, side), fill)
    out.paste(img, ((side - w) // 2, (side - h) // 2))   # ortala: dolgu iki yana eşit
    return out