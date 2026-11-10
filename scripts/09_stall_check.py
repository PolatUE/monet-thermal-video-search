import numpy as np
import pandas as pd

from monet.config import OUTPUT_DIR
from monet.dataset import TELEMETRY_KEYS

SRC_FPS = 25.0
SMOOTH_FRAMES = 25    # saniye yuvarlamasından gelen testere dişini yumuşatmak için
HALF_WIN_SEC = 5.0    # basamağın iki yanında bakılacak süre (A = -5 sn, B = +5 sn)
STEP_MIN_SEC = 1.0    # 10 sn'de bundan fazla artan fark = basamak
CONTROL_EVERY_SEC = 30.0  # yöntem kontrolü: basamaksız yerlerde her 30 sn'de bir ölçüm


def haversine_m(lat1, lon1, lat2, lon2):
    """İki GPS noktası arası yer yüzeyindeki mesafe (metre)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def implied(df, vt, ts, a, b):
    """A ile B kareleri arası: kare süresi, saat süresi, GPS yolu, ortalama hız, ima edilen süre."""
    seg = df.iloc[a:b + 1]
    path = haversine_m(seg["latitude"].values[:-1], seg["longitude"].values[:-1],
                       seg["latitude"].values[1:], seg["longitude"].values[1:]).sum()
    v = (seg["speed_x"] / 100).mean()  # 3b K5: speed_x = cm/sn
    return vt[b] - vt[a], ts[b] - ts[a], path, v, path / v


def check_sequence(seq, df):
    df = df.sort_values("frame_no").reset_index(drop=True)
    n = df["frame_no"].to_numpy()
    vt = (n - n[0]) / SRC_FPS                                     # kare numarasından süre
    ts = (df["t"] - df["t"].iloc[0]).dt.total_seconds().to_numpy()  # saatten süre
    off = pd.Series(ts - vt).rolling(SMOOTH_FRAMES, center=True, min_periods=1).mean().to_numpy()

    # Her karede: 5 sn sonrasındaki fark - 5 sn öncesindeki fark
    jump = np.interp(vt + HALF_WIN_SEC, vt, off) - np.interp(vt - HALF_WIN_SEC, vt, off)
    inside = (vt >= HALF_WIN_SEC) & (vt <= vt[-1] - HALF_WIN_SEC)
    is_step = (jump > STEP_MIN_SEC) & inside

    rows = []
    # Ardışık işaretli kareleri tek basamak say; merkezi = sıçramanın en büyük olduğu kare
    region = np.cumsum(np.r_[True, np.diff(is_step.astype(int)) != 0])
    for r in np.unique(region[is_step]):
        idx = np.where((region == r) & is_step)[0]
        c = idx[np.argmax(jump[idx])]
        a = np.searchsorted(vt, vt[c] - HALF_WIN_SEC)
        b = np.searchsorted(vt, vt[c] + HALF_WIN_SEC)
        d_frame, d_clock, path, v, imp = implied(df, vt, ts, a, b)
        verdict = "(ii) saat doğru" if abs(imp - d_clock) < abs(imp - d_frame) else "(i) kare no doğru"
        rows.append({"seq": seq, "video_sn": round(vt[c], 1),
                     "A": df["file_name"].iloc[a], "B": df["file_name"].iloc[b],
                     "kare_sure": round(d_frame, 1), "saat_sure": round(d_clock, 1),
                     "gps_yol_m": round(path, 1), "hiz_mps": round(v, 2),
                     "ima_sure": round(imp, 1), "sonuc": verdict})

    # Yöntem kontrolü: basamak olmayan yerlerde ima edilen süre / saat süresi ≈ 1 olmalı
    ratios = []
    for center in np.arange(CONTROL_EVERY_SEC, vt[-1] - HALF_WIN_SEC, CONTROL_EVERY_SEC):
        c = np.searchsorted(vt, center)
        if abs(jump[c]) > 0.3:          # yakında basamak varsa kontrol için kullanma
            continue
        a = np.searchsorted(vt, vt[c] - HALF_WIN_SEC)
        b = np.searchsorted(vt, vt[c] + HALF_WIN_SEC)
        _, d_clock, _, v, imp = implied(df, vt, ts, a, b)
        if v > 1.0:                      # neredeyse durağan drone'da oran anlamsız
            ratios.append(imp / d_clock)
    control = {"seq": seq, "kontrol_sayisi": len(ratios),
               "ima/gercek_medyan": round(float(np.median(ratios)), 3) if ratios else np.nan}
    return rows, control


frames = pd.read_parquet(OUTPUT_DIR / "frames.parquet")
tel = frames[frames[TELEMETRY_KEYS].notna().all(axis=1)]  # yalnızca telemetrili sekanslar
steps, controls = [], []
for seq, g in tel.groupby("seq"):
    r, c = check_sequence(seq, g)
    steps += r
    controls.append(c)

steps = pd.DataFrame(steps)
out_dir = OUTPUT_DIR / "suitability"
out_dir.mkdir(parents=True, exist_ok=True)
steps.to_csv(out_dir / "stall_check.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 220)
print("== Basamaklar ==")
print(steps.to_string(index=False) if len(steps) else "Basamak bulunmadı")
print("\n== Yöntem kontrolü (basamaksız yerler; 1'e yakın olmalı) ==")
print(pd.DataFrame(controls).to_string(index=False))