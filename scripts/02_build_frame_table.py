import pandas as pd

from monet.config import OUTPUT_DIR
from monet.data.dataset import (TELEMETRY_KEYS, annotations_table, find_sequences,
                           frames_table, load_labels)

# Her labels.json yalnızca bir kez okunur; iki tablo da aynı okumadan üretilir
labels = {name: load_labels(d) for name, d in find_sequences().items()}
frames = pd.concat([frames_table(n, lb) for n, lb in labels.items()], ignore_index=True)
anns = pd.concat([annotations_table(n, lb) for n, lb in labels.items()], ignore_index=True)

# Çıktılar ham verinin dışında, outputs/ altında
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
frames.to_parquet(OUTPUT_DIR / "frames.parquet", index=False)
anns.to_parquet(OUTPUT_DIR / "annotations.parquet", index=False)
print(f"Yazıldı: frames.parquet ({len(frames)} kare), annotations.parquet ({len(anns)} kutu)\n")

# Sekans başına özet: sonraki adımların varsayımlarını kontrol etmek için
g = frames.groupby("seq")
summary = pd.DataFrame({
    "kare": g.size(),
    "ilk_no": g["frame_no"].min(),
    "son_no": g["frame_no"].max(),
    "tekrar_no": g["frame_no"].apply(lambda s: s.duplicated().sum()),
    "hep_tek": g["frame_no"].apply(lambda s: bool((s % 2 == 1).all())),
    "sure_sn": g["t"].apply(lambda s: (s.max() - s.min()).total_seconds()),
    "telemetri_%": g[TELEMETRY_KEYS].apply(lambda x: round(100 * x.notna().to_numpy().mean(), 1)),
})
# Sınıf başına kutu sayısı: bağlam dosyasındaki tabloyla karşılaştırmak için
summary = summary.join(anns.groupby(["seq", "category"]).size().unstack(fill_value=0))
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", None)
print(summary)