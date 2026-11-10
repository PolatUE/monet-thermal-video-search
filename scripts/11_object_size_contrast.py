import time

import numpy as np
import pandas as pd
from PIL import Image

from monet.config import OUTPUT_DIR, RAW_DIR

# Onaylı varsayım ve eşikler
FRAME_W, FRAME_H = 800, 600
MODEL_PX, PATCH_PX = 224, 14   # model girişi ve yama boyutu (Faz 3'te model seçilince güncellenir)
TILINGS = [1, 2, 3, 4]         # 1x1 (tiling yok), 2x2, 3x3, 4x4
VISIBLE_MIN = 0.80             # K3: 2x2'de görünür olması gereken kutu oranı
EDGE_PX = 100                  # K4: kare ortadan kırpılınca her iki yandan kaybolan şerit
RING_PX = 5                    # K5: kutunun çevresindeki halka kalınlığı
LOW_CONTRAST = 10              # K5: iç ve halka parlaklık farkı bunun altındaysa düşük kontrast
LOW_MAX = 0.10                 # K5: düşük kontrastlı kutu oranı üst sınırı
SAMPLE_EVERY = 5               # K5: her 5. kare okunur
CLASSES = ["person", "vehicle"]


def min_tiling(size: pd.Series) -> pd.Series:
    """Kutunun en az 1 yama büyüklüğünde göründüğü en küçük tiling (1..4); hiçbirinde değilse 99."""
    out = pd.Series(99, index=size.index)
    for k in reversed(TILINGS):  # büyükten küçüğe: en küçük uygun k en son yazılır
        scale = MODEL_PX / (FRAME_H / k)  # parça yüksekliği 600/k, modele 224 px olarak girer
        out[size * scale >= PATCH_PX] = k
    return out


def box_contrast(gray: np.ndarray, x, y, w, h) -> float:
    """Kutu içi ortalama parlaklık ile çevresindeki 5 px halkanın ortalaması arasındaki fark."""
    H, W = gray.shape
    x0, y0 = max(int(x), 0), max(int(y), 0)
    x1, y1 = min(int(np.ceil(x + w)), W), min(int(np.ceil(y + h)), H)
    if x1 <= x0 or y1 <= y0:
        return np.nan
    X0, Y0 = max(x0 - RING_PX, 0), max(y0 - RING_PX, 0)
    X1, Y1 = min(x1 + RING_PX, W), min(y1 + RING_PX, H)
    outer = gray[Y0:Y1, X0:X1].astype(float)
    inner = gray[y0:y1, x0:x1].astype(float)
    ring_n = outer.size - inner.size
    if ring_n == 0:
        return np.nan
    ring_mean = (outer.sum() - inner.sum()) / ring_n  # halka = dış dikdörtgen - kutu
    return abs(inner.mean() - ring_mean)


frames = pd.read_parquet(OUTPUT_DIR / "frames.parquet")
anns = pd.read_parquet(OUTPUT_DIR / "annotations.parquet")
boxes = anns[anns["category"].isin(CLASSES)].merge(
    frames[["seq", "image_id", "frame_no", "file_name"]], on=["seq", "image_id"])

# K3: boyut ve gereken tiling
boxes["size"] = np.sqrt(boxes["w"] * boxes["h"])  # ortalama kenar
boxes["min_tiling"] = min_tiling(boxes["size"])

# K4: kare ortadan kare biçimine kırpılırsa tamamen dışarıda kalan kutular
boxes["edge"] = (boxes["x"] + boxes["w"] <= EDGE_PX) | (boxes["x"] >= FRAME_W - EDGE_PX)

# K5: her sekansta her 5. karenin kutularında kontrast (PNG yalnızca okunur)
boxes["contrast"] = np.nan
t0 = time.perf_counter()
for seq, f in frames.sort_values("frame_no").groupby("seq"):
    sample = f.iloc[::SAMPLE_EVERY]
    sb = boxes[(boxes["seq"] == seq) & boxes["image_id"].isin(sample["image_id"])]
    for file_name, bx in sb.groupby("file_name"):
        with Image.open(RAW_DIR / seq / "lwir" / file_name) as im:
            gray = np.asarray(im.convert("L"))  # görüntüler gri; tek kanal yeterli
        boxes.loc[bx.index, "contrast"] = [box_contrast(gray, r.x, r.y, r.w, r.h)
                                           for r in bx.itertuples()]
    print(f"  {seq:12s} {len(sample):5d} kare okundu  ({time.perf_counter() - t0:5.1f} sn)")
boxes["low_contrast"] = boxes["contrast"] < LOW_CONTRAST

out_dir = OUTPUT_DIR / "suitability"
out_dir.mkdir(parents=True, exist_ok=True)
boxes.to_parquet(out_dir / "box_stats.parquet", index=False)  # 3d'de kutu bazında kullanılacak

# Özet tablolar
rows, verdict = [], {}
for c in CLASSES:
    b = boxes[boxes["category"] == c]
    s = b["contrast"].dropna()
    vis = {k: round(100 * (b["min_tiling"] <= k).mean(), 1) for k in TILINGS}
    low = (s < LOW_CONTRAST).mean() if len(s) else np.nan
    rows.append({
        "sinif": c, "kutu": len(b), "boyut_medyan_px": round(b["size"].median(), 1),
        "gorunur_1x1_%": vis[1], "gorunur_2x2_%": vis[2],
        "gorunur_3x3_%": vis[3], "gorunur_4x4_%": vis[4],
        "kenarda_%": round(100 * b["edge"].mean(), 1),
        "kontrast_olculen": len(s), "kontrast_medyan": round(s.median(), 1),
        "dusuk_kontrast_%": round(100 * low, 1),
    })
    verdict[f"K3_{c}"] = "uygun" if vis[2] / 100 >= VISIBLE_MIN else "sorunlu"
    verdict[f"K5_{c}"] = "uygun" if low <= LOW_MAX else "sorunlu"

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
print("\n== Sınıf bazında (K3 boyut, K4 kenar, K5 kontrast) ==")
print(pd.DataFrame(rows).set_index("sinif").T)

print("\n== Sekans bazında düşük kontrast oranı (%) ==")
sampled = boxes.dropna(subset=["contrast"])
print((100 * sampled.groupby(["seq", "category"])["low_contrast"].mean()).round(1).unstack())

print("\n== Kararlar ==")
print(pd.Series(verdict).to_string())