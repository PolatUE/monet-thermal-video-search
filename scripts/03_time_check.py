import numpy as np
import pandas as pd

from monet.config import OUTPUT_DIR

# Onaylanan eşikler (tek yerde; değiştirmek gerekirse yalnızca burası)
STANDARD_FPS = [25.0, 29.97, 30.0]  # yaygın kamera hızları
FPS_TOL = 0.02                       # Kontrol 2 ve 3: %2
SEG_SEC = 5.0                        # çalışma segment uzunluğu
MIN_REAL_PER_SEG = 8                 # Kontrol 5: segmentte en az gerçek kare
MAX_GAP_SEC = 5.0                    # Kontrol 5: en uzun boşluk


def longest_run(mask: np.ndarray) -> int:
    """Art arda gelen en uzun True dizisinin uzunluğu."""
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def check_sequence(df: pd.DataFrame) -> tuple[dict, dict]:
    df = df.sort_values("frame_no")
    n = df["frame_no"].to_numpy()
    # Kayıttaki zaman, ilk kareden itibaren saniye
    ts = (df["t"] - df["t"].iloc[0]).dt.total_seconds().to_numpy()

    # Kontrol 1: kare sırasıyla giderken zaman geri gidiyor mu?
    back = int((np.diff(ts) < 0).sum())

    # Kontrol 2: kare_no = fps * zaman + b  (eğim = kamera fps'i)
    fps, b = np.polyfit(ts, n, 1)
    r2 = 1 - ((n - (fps * ts + b)) ** 2).sum() / ((n - n.mean()) ** 2).sum()
    std = min(STANDARD_FPS, key=lambda f: abs(fps - f))
    dev = abs(fps - std) / std
    fps_ends = (n[-1] - n[0]) / ts[-1]  # yalnızca uç noktalardan, karşılaştırma için

    # Kontrol 3: zamana göre üç eşit parça, her birinde ayrı fps
    edges = np.linspace(0, ts[-1], 4)
    part_fps = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ts >= lo) & (ts <= hi)
        part_fps.append(np.polyfit(ts[m], n[m], 1)[0])
    spread = (max(part_fps) - min(part_fps)) / np.mean(part_fps)

    # Kontrol 4: artık = kayıttaki zaman - kare numarasından beklenen zaman
    resid = ts - (n - b) / fps
    frac_1s = np.mean(np.abs(resid) <= 1)
    run_2s = longest_run(np.abs(resid) > 2)

    # Kontrol 5a: ardışık iki gerçek kare arası süre (saniye)
    gaps = np.diff(n) / fps
    n_gap_2_5 = int(((gaps > 2) & (gaps <= MAX_GAP_SEC)).sum())

    # Kontrol 5b: video zamanına göre 5 sn'lik segmentlerde gerçek kare sayısı
    vt = (n - n[0]) / fps
    seg = np.floor(vt / SEG_SEC).astype(int)
    n_full = int(vt[-1] // SEG_SEC)                      # son yarım segment sayılmaz
    counts = np.bincount(seg, minlength=n_full + 1)[:n_full]
    seg_ok = np.mean(counts >= MIN_REAL_PER_SEG) if n_full else 0.0

    v = {
        "K1": "uygun" if back == 0 else "sorunlu",
        "K2": "uygun" if dev <= FPS_TOL else "belirsiz",
        "K3": "uygun" if spread <= FPS_TOL else "sorunlu",
        "K4": "uygun" if frac_1s >= 0.99 and run_2s < 2 else "sorunlu",
        "K5": "uygun" if gaps.max() <= MAX_GAP_SEC and seg_ok >= 0.95 else "sorunlu",
    }
    vals = list(v.values())
    v["SONUC"] = "sorunlu" if "sorunlu" in vals else ("belirsiz" if "belirsiz" in vals else "uygun")

    metrics = {
        "geri_gitme": back,
        "fps_reg": round(fps, 3),
        "fps_uc": round(fps_ends, 3),
        "R2": round(r2, 6),
        "std_sapma_%": round(100 * dev, 2),
        "parca_fps": " / ".join(f"{p:.2f}" for p in part_fps),
        "parca_fark_%": round(100 * spread, 2),
        "artik_le1s_%": round(100 * frac_1s, 2),
        "max_artik_sn": round(np.abs(resid).max(), 2),
        "art_arda_gt2s": run_2s,
        "max_bosluk_sn": round(gaps.max(), 2),
        "bosluk_2_5sn": n_gap_2_5,
        "segment": n_full,
        "seg_ge8_%": round(100 * seg_ok, 1),
        "seg_min_kare": int(counts.min()) if n_full else 0,
    }
    return metrics, v


frames = pd.read_parquet(OUTPUT_DIR / "frames.parquet")
results = {seq: check_sequence(g) for seq, g in frames.groupby("seq")}
metrics = pd.DataFrame({s: r[0] for s, r in results.items()}).T
verdicts = pd.DataFrame({s: r[1] for s, r in results.items()}).T

out_dir = OUTPUT_DIR / "suitability"
out_dir.mkdir(parents=True, exist_ok=True)
metrics.join(verdicts).to_csv(out_dir / "time_check.csv", encoding="utf-8-sig")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print("== Ölçümler ==")
print(metrics.T)   # sekanslar sütunda: yan yana karşılaştırmak kolay
print("\n== Kararlar ==")
print(verdicts)