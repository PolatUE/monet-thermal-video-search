"""Puan tablosu: sorgu dosyasındaki her ifadeyi bir indeks üzerinde ölçer, outputs/eval/scores.csv'ye ekler.

Örnek: python scripts/evaluate.py --label v0
"""
import argparse
from datetime import datetime, timezone

import pandas as pd

from monet.config import OUTPUT_DIR
from monet.eval import metrics as M
from monet.eval.ground_truth import load_queries, relevance, segment_labels
from monet.search.engine import DEFAULT_INDEX, SearchEngine

SCORES_PATH = OUTPUT_DIR / "eval" / "scores.csv"


def evaluate(engine: SearchEngine, queries: dict, labels: pd.DataFrame) -> pd.DataFrame:
    # Doğru cevapları motorun segment sırasına hizala: lab'ın i. satırı = vektörlerin i. satırı
    lab = engine.meta[["clip", "seg"]].merge(labels, on=["clip", "seg"], how="left", validate="one_to_one")
    rows = []
    for item in queries["queries"]:
        r = relevance(lab, item["target"])
        rel, group, seq = r["rel"], r["group"], lab["seq"]
        b10, b20 = M.random_baseline(rel, 10), M.random_baseline(rel, 20)
        rows.append({"query": item["id"], "text": "(rastgele taban)",
                     "P@10": b10["precision"], "R@10": b10["recall"], "R@20": b20["recall"],
                     "AUC": 0.5, "AUC_sekans_ici": 0.5,
                     "R@20_buyuk": b20["recall"], "R@20_kucuk": b20["recall"]})
        for text in item["texts"]:
            s = pd.Series(engine.scores(text), index=lab.index)
            w_auc, n_seq = M.within_group_auc(s, rel, seq)
            top10 = s.sort_values(ascending=False).index[:10]       # kullanıcının gördüğü ilk 10
            rows.append({
                "query": item["id"], "text": text,
                "P@10": M.precision_at_k(s, rel, 10), "R@10": M.recall_at_k(s, rel, 10),
                "R@20": M.recall_at_k(s, rel, 20), "AUC": M.auc(s, rel),
                "AUC_sekans_ici": w_auc, "sekans_sayisi": n_seq,
                "R@20_buyuk": M.subgroup_recall_at_k(s, rel, group, "big", 20),
                "R@20_kucuk": M.subgroup_recall_at_k(s, rel, group, "small", 20),
                "farkli_klip_ilk10": lab.loc[top10, "clip"].nunique(),
            })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Puan tablosuna satır ekle")
    p.add_argument("--label", required=True, help="deney adı, ör. v0")
    p.add_argument("--index", default=DEFAULT_INDEX, help="indeks klasör adı")
    args = p.parse_args()

    queries = load_queries()
    engine = SearchEngine(args.index)
    res = evaluate(engine, queries, segment_labels())

    # Satırın hangi koşulda üretildiği: deney, indeks, sorgu dosyası sürümü, tarih
    res.insert(0, "label", args.label)
    res.insert(1, "index", args.index)
    res.insert(2, "queries_version", queries["version"])
    res["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(SCORES_PATH, mode="a", header=not SCORES_PATH.exists(), index=False, encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 45)
    cols = ["query", "text", "P@10", "R@10", "R@20", "AUC", "AUC_sekans_ici",
            "R@20_buyuk", "R@20_kucuk", "farkli_klip_ilk10"]
    print(res[cols].round(3).to_string(index=False))
    print(f"\neklendi: {SCORES_PATH} ({len(res)} satir)")


if __name__ == "__main__":
    main()