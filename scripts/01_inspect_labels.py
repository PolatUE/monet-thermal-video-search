import sys
from collections import Counter

from monet.data.dataset import find_sequences, load_labels


def describe(obj, name="root", depth=0, max_depth=4):
    """JSON ağacını gezer: sözlüklerde anahtarları, listelerde uzunluğu ve İLK elemanın yapısını yazar."""
    pad = "  " * depth
    if isinstance(obj, dict):
        print(f"{pad}{name}: dict ({len(obj)} anahtar)")
        if depth < max_depth:
            for k, v in obj.items():
                describe(v, k, depth + 1, max_depth)
    elif isinstance(obj, list):
        print(f"{pad}{name}: list ({len(obj)} eleman)")
        if obj and depth < max_depth:
            describe(obj[0], "[0]", depth + 1, max_depth)
    else:
        line = f"{pad}{name}: {type(obj).__name__} = {obj!r}"
        print(line[:120])  # uzun değerleri kırp, ekran taşmasın


def key_variants(records, label):
    """İlk kayıt temsilî mi? Tüm kayıtlardaki farklı anahtar kümelerini sayar."""
    c = Counter(
        tuple(sorted(r.keys())) if isinstance(r, dict) else type(r).__name__
        for r in records
    )
    print(f"{label}: {len(c)} farklı yapı")
    for keys, n in c.most_common(3):
        print(f"  {n} kayıt -> {keys}")


seqs = find_sequences()
print("Bulunan sekanslar:", ", ".join(seqs))

# Varsayılan dirtroad_04: hem telemetrisi hem iki sınıf etiketi var
name = sys.argv[1] if len(sys.argv) > 1 else "dirtroad_04"
if name not in seqs:
    sys.exit(f"'{name}' bulunamadı. Seçenekler: {list(seqs)}")
seq_dir = seqs[name]
labels = load_labels(seq_dir)

print(f"\n== {name}: alt klasörler ==")
print([p.name for p in seq_dir.iterdir() if p.is_dir()])

print("\n== JSON yapısı (ilk elemanlar) ==")
describe(labels, "labels.json")

print("\n== Yapı tutarlılığı (tüm kayıtlar) ==")
images = labels.get("images", [])
anns = labels.get("annotations", [])
key_variants(images, "images")
key_variants([im.get("metadata") for im in images], "images.metadata")
key_variants(anns, "annotations")
key_variants([a.get("attributes") for a in anns], "annotations.attributes")

print("\n== Dosya eşleşmesi ==")
lwir = seq_dir / "lwir"
n_png = len(list(lwir.glob("*.png"))) if lwir.is_dir() else 0
print(f"lwir klasöründe {n_png} PNG, labels.json'da {len(images)} görüntü kaydı")
# file_name'in hangi klasöre göre yazıldığını bilmiyoruz; iki adayı da deneriz
for base in (seq_dir, lwir):
    ok = sum((base / im["file_name"]).is_file() for im in images)
    print(f"  {base.name}/ + file_name -> {ok}/{len(images)} dosya bulundu")