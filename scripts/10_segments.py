import numpy as np
import pandas as pd

from monet.config import OUTPUT_DIR

SEG_SEC = 5.0
# Telemetrisiz sekanslarda güneş hesabı için MONET bölgesinin yaklaşık konumu
DEFAULT_LAT, DEFAULT_LON = 35.095, 33.079
MIN_SHARE = 0.10             # K1: bir bandın sayılması için en az pay
MIN_SEG, MIN_PCT = 20, 0.05  # K2: var ve yok için en az segment sayısı ve payı
CLASSES = ["person", "vehicle"]


def sun_elevation_deg(t: pd.Series, lat, lon) -> np.ndarray:
    """Güneşin ufuktan yüksekliği (derece). Astronomik almanağın kısa formülü, ~0,01° hassas."""
    n = (t - pd.Timestamp("2000-01-01 12:00", tz="UTC")).dt.total_seconds().to_numpy() / 86400
    L = (280.460 + 0.9856474 * n) % 360                       # güneşin ortalama boylamı
    g = np.radians((357.528 + 0.9856003 * n) % 360)           # ortalama anomali
    lam = np.radians(L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.radians(23.439 - 0.0000004 * n)                  # dünya ekseninin eğikliği
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    gmst = np.radians((280.46061837 + 360.98564736629 * n) % 360)
    ha = gmst + np.radians(lon) - ra                          # saat açısı
    la = np.radians(lat)
    return np.degrees(np.arcsin(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)))


def band(s: pd.Series, lo, hi, names, hi_incl=False):
    """s < lo -> names[0]; lo..hi -> names[1]; üstü -> names[2]; eksik değer -> None."""
    mid = (s <= hi) if hi_incl else (s < hi)
    out = np.select([s < lo, mid], names[:2], names[2]).astype(object)
    out[s.isna().to_numpy()] = None
    return out


clips = pd.read_parquet(OUTPUT_DIR / "clips.parquet")
slots = pd.read_parquet(OUTPUT_DIR / "slots.parquet")
frames = pd.read_parquet(OUTPUT_DIR / "frames.parquet")
anns = pd.read_parquet(OUTPUT_DIR / "annotations.parquet")

# 1) Her video karesine segment numarası; son yarım segment dahil edilmez
slots = slots.merge(clips[["clip", "seq", "sure_sn"]], on="clip")
slots["seg"] = np.floor(slots["video_sec"] / SEG_SEC).astype(int)
slots = slots[slots["seg"] < np.floor(slots["sure_sn"] / SEG_SEC)]

# 2) Gerçek kareler: telemetri ve kare başına sınıf sayıları
per_frame = anns.groupby(["seq", "image_id", "category"]).size().unstack(fill_value=0)
real = (slots[slots["is_real"]]
        .merge(frames[["seq", "frame_no", "image_id", "altitude", "gimbal_pitch",
                       "latitude", "longitude"]], on=["seq", "frame_no"])
        .merge(per_frame, left_on=["seq", "image_id"], right_index=True, how="left"))
for c in per_frame.columns:
    real[c] = real[c].fillna(0).astype(int)

# 3) Segment tablosu: bir satır = bir 5 sn'lik segment
base = slots.groupby(["clip", "seg"]).agg(
    seq=("seq", "first"), video_bas=("video_sec", "min"), yuva=("slot", "size"))
g = real.groupby(["clip", "seg"])
stats = g.agg(
    gercek=("slot", "size"), t_bas=("t_src", "min"), t_son=("t_src", "max"),
    altitude=("altitude", "median"), gimbal_pitch=("gimbal_pitch", "median"),
    lat=("latitude", "median"), lon=("longitude", "median"))
for c in per_frame.columns:  # sınıfın göründüğü gerçek kare sayısı ve karedeki en fazla kutu
    stats[f"{c}_kare"] = g[c].apply(lambda s: int((s > 0).sum()))
    stats[f"{c}_max"] = g[c].max()
seg = base.join(stats, how="left").reset_index()
seg["gercek"] = seg["gercek"].fillna(0).astype(int)

# 4) Güneş yüksekliği: segmentin ortasındaki gerçek zaman ve konum
t_mid = seg["t_bas"] + (seg["t_son"] - seg["t_bas"]) / 2
seg["sun_elev"] = sun_elevation_deg(t_mid, seg["lat"].fillna(DEFAULT_LAT),
                                    seg["lon"].fillna(DEFAULT_LON)).round(2)

# 5) Bantlar (onaylı sınırlar)
seg["b_irtifa"] = band(seg["altitude"], 60, 100, ["alçak <60", "orta 60-100", "yüksek >=100"])
seg["b_aci"] = band(seg["gimbal_pitch"], -45, -30, ["dik <-45", "orta -45..-30", "sığ >-30"], hi_incl=True)
seg["b_gunes"] = band(seg["sun_elev"], 0, 10, ["batım sonrası <0", "alçak 0-10", "yüksek >10"], hi_incl=True)
for c in CLASSES:
    seg[f"has_{c}"] = seg.get(f"{c}_kare", 0) > 0

seg.to_parquet(OUTPUT_DIR / "segments.parquet", index=False)
print(f"Yazıldı: segments.parquet ({len(seg)} segment, {seg['clip'].nunique()} klip)\n")

# K1: sahne çeşitliliği
rows, verdict = [], {}
for dim, col in [("irtifa", "b_irtifa"), ("kamera_acisi", "b_aci"), ("gunes", "b_gunes")]:
    vc = seg[col].value_counts()
    share = vc / vc.sum()
    for b, k in vc.items():
        rows.append({"boyut": dim, "bant": b, "segment": k, "pay_%": round(100 * share[b], 1)})
    verdict[dim] = "uygun" if (share >= MIN_SHARE).sum() >= 2 else "sorunlu"
print("== K1: Sahne çeşitliliği (ölçülebilen segmentler) ==")
print(pd.DataFrame(rows).to_string(index=False))

# K2: nesne var / yok
rows = []
for c in CLASSES:
    var = int(seg[f"has_{c}"].sum())
    yok = len(seg) - var
    ok = min(var, yok) >= MIN_SEG and min(var, yok) / len(seg) >= MIN_PCT
    rows.append({"sinif": c, "var": var, "yok": yok,
                 "var_%": round(100 * var / len(seg), 1), "yok_%": round(100 * yok / len(seg), 1)})
    verdict[f"{c}_var_yok"] = "uygun" if ok else "sorunlu"
print("\n== K2: Nesne var / yok ==")
print(pd.DataFrame(rows).to_string(index=False))

# Sekans bazında özet: bantların ve sınıfların nereden geldiğini görmek için
print("\n== Sekans bazında ==")
summ = seg.groupby("seq").agg(
    segment=("seg", "size"), irtifa_medyan=("altitude", "median"),
    aci_medyan=("gimbal_pitch", "median"), gunes_min=("sun_elev", "min"),
    gunes_max=("sun_elev", "max"),
    person_var=("has_person", "sum"), vehicle_var=("has_vehicle", "sum"))
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print(summ)

print("\n== Kararlar ==")
print(pd.Series(verdict).to_string())