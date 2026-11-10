import numpy as np
import pandas as pd

from monet.config import OUTPUT_DIR
from monet.dataset import TELEMETRY_KEYS

# Onaylanan eşikler (tek yerde)
SRC_FPS = 25.0
FILL_MIN = 0.95          # K1: doluluk
HZ_MIN = 1.0             # K2: ortalama güncelleme sıklığı
FREEZE_MAX_SEC = 2.0     # K2: en uzun donma
OUT_MAX = 0.01           # K3: aralık dışı oranı
RADIUS_M = 5000          # K3: ilk kareden en fazla uzaklık
SPEED_MAX = 30.0         # K4: m/sn
SPEED_OK_MIN = 0.99      # K4: makul hızdaki aralık oranı
RATIO_TOL = 0.25         # K5: oranın dağınıklığı (çeyrekler arası / medyan)
RATIO_WIN_SEC= 5.0

# K3: alan -> (alt, üst) kabul aralığı
RANGES = {
    "altitude": (0, 500),
    "gimbal_pitch": (-90, 30),
    "UAV_yaw": (0, 360), "gimbal_yaw": (-180, 180),#dronea göre kamera açısı
    "UAV_roll": (-90, 90), "UAV_pitch": (-90, 90), "gimbal_roll": (-90, 90),
}


def haversine_m(lat1, lon1, lat2, lon2):
    """İki GPS noktası arası yer yüzeyindeki mesafe (metre)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def check_sequence(df: pd.DataFrame):
    df = df.sort_values("frame_no").reset_index(drop=True)
    fill = df[TELEMETRY_KEYS].notna().to_numpy().mean()
    if fill == 0:
        return {"doluluk_%": 0.0}, {"SONUC": "telemetrisiz"}

    # Zaman: kare numarasından (saniye altı hassas). Kayıt takılmaları K6'da ayrıca incelenecek.
    tf = (df["frame_no"] - df["frame_no"].iloc[0]).to_numpy() / SRC_FPS
    tel = df[TELEMETRY_KEYS].to_numpy()

    # K2: 12 alandan herhangi biri bir önceki kareden farklıysa = güncelleme
    changed = np.r_[True, (tel[1:] != tel[:-1]).any(axis=1)]
    hz = (changed.sum() - 1) / tf[-1]
    # Donma: değerlerin aynı kaldığı ardışık kare dizisinin süresi (kare boşlukları sayılmaz)
    run_id = np.cumsum(changed)
    freeze = pd.Series(tf).groupby(run_id).agg(lambda s: s.iloc[-1] - s.iloc[0]).max()

    # K3: alan bazında aralık dışı oranı + ilk kareden uzaklık
    out = {k: float(((df[k] < lo) | (df[k] > hi)).mean()) for k, (lo, hi) in RANGES.items()}
    dist0 = haversine_m(df["latitude"].iloc[0], df["longitude"].iloc[0],
                        df["latitude"], df["longitude"])
    out["konum"] = float((dist0 > RADIUS_M).mean())
    worst = max(out, key=out.get)

    # K4: 1 sn'lik aralıkların ilk karesini al, ardışık aralıklar arası GPS hızı
    w = df.assign(tf=tf).groupby(np.floor(tf)).first()
    dt = np.diff(w["tf"].to_numpy())
    d = haversine_m(w["latitude"].to_numpy()[:-1], w["longitude"].to_numpy()[:-1],
                    w["latitude"].to_numpy()[1:], w["longitude"].to_numpy()[1:])
    keep = dt <= 2.0                      # kare boşluğu olan aralıkları hesaba katma
    gps_v = d[keep] / dt[keep]
    v_ok = np.mean(gps_v <= SPEED_MAX)

    # K5: 5 sn'lik pencerelerde ortalama hız alanı / pencere boyunca GPS'ten ölçülen hız
    #     (uzun pencere GPS gürültüsünü azaltır; ortalama, anlık oynamayı yumuşatır)
    win = df.assign(tf=tf, h=np.hypot(df["speed_x"], df["speed_y"])).groupby(
        np.floor(tf / RATIO_WIN_SEC))
    first5 = win.first()
    dt5 = np.diff(first5["tf"].to_numpy())
    d5 = haversine_m(first5["latitude"].to_numpy()[:-1], first5["longitude"].to_numpy()[:-1],
                     first5["latitude"].to_numpy()[1:], first5["longitude"].to_numpy()[1:])
    gps5 = d5 / dt5
    sp5 = win["h"].mean().to_numpy()[:-1]
    ok5 = (dt5 <= 1.5 * RATIO_WIN_SEC) & (gps5 > 1.0)  # boşluklu ve neredeyse durağan pencereleri atla
    ratio = sp5[ok5] / gps5[ok5]
    if len(ratio) >= 10:
        q25, med, q75 = np.percentile(ratio, [25, 50, 75])
        spread = (q75 - q25) / med
    else:
        med, spread = np.nan, np.nan

    v = {
        "K1": "uygun" if fill >= FILL_MIN else "sorunlu",
        "K2": "uygun" if hz >= HZ_MIN and freeze <= FREEZE_MAX_SEC else "sorunlu",
        "K3": "uygun" if out[worst] <= OUT_MAX else "sorunlu",
        "K4": "uygun" if v_ok >= SPEED_OK_MIN else "sorunlu",
        "K5": "uygun" if spread <= RATIO_TOL else "belirsiz",  # NaN ise de belirsiz
    }
    vals = list(v.values())
    v["SONUC"] = "sorunlu" if "sorunlu" in vals else ("belirsiz" if "belirsiz" in vals else "uygun")

    m = {
        "doluluk_%": round(100 * fill, 1),
        "guncelleme_hz": round(hz, 2),
        "en_uzun_donma_sn": round(freeze, 2),
        "en_kotu_alan": f"{worst} %{100 * out[worst]:.2f}",
        "gps_hiz_medyan": round(np.median(gps_v), 2),
        "gps_hiz_max": round(gps_v.max(), 1),
        "hiz_le30_%": round(100 * v_ok, 2),
        "hiz_oran_medyan": round(med, 1),
        "hiz_oran_daginiklik": round(spread, 3),
    }
    return m, v


frames = pd.read_parquet(OUTPUT_DIR / "frames.parquet")
results = {seq: check_sequence(g) for seq, g in frames.groupby("seq")}
metrics = pd.DataFrame({s: r[0] for s, r in results.items()}).T
verdicts = pd.DataFrame({s: r[1] for s, r in results.items()}).T

out_dir = OUTPUT_DIR / "suitability"
out_dir.mkdir(parents=True, exist_ok=True)
metrics.join(verdicts).to_csv(out_dir / "telemetry_check.csv", encoding="utf-8-sig")

# Alan bazında değer aralığı: aykırıların nereden geldiğini görmek ve 3c için
has_tel = frames[frames[TELEMETRY_KEYS].notna().any(axis=1)]
ranges = has_tel.groupby("seq")[TELEMETRY_KEYS].agg(["min", "max"]).T

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print("== Ölçümler ==")
print(metrics.T.fillna("-"))
print("\n== Kararlar ==")
print(verdicts.fillna("-"))
print("\n== Alan aralıkları (telemetrili sekanslar) ==")
print(ranges)