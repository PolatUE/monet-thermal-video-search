"""Puan tablosu: sorgu dosyasındaki her ifadeyi bir indeks üzerinde ölçer, outputs/eval/scores.csv'ye ekler.

Filtreli (hibrit) sorgular: her filtre için rastgele ve filtre + rastgele taban, her ifade için
yalnız vektör ve hibrit satırı. Doğru cevap = nesne var VE filtre koşulu doğru.

Örnek: python scripts/evaluate.py --label v1
       python scripts/evaluate.py --label v1 --dry-run   (tabloya yazmaz, yalnızca ekrana basar)
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
        rows.append({"query": item["id"], "yontem": "rastgele", "text": "(rastgele taban)",
                     "P@10": b10["precision"], "R@10": b10["recall"], "R@20": b20["recall"],
                     "AUC": 0.5, "AUC_sekans_ici": 0.5,
                     "R@20_buyuk": b20["recall"], "R@20_kucuk": b20["recall"]})
        for text in item["texts"]:
            s = pd.Series(engine.scores(text), index=lab.index)
            w_auc, n_seq = M.within_group_auc(s, rel, seq)
            top10 = s.sort_values(ascending=False).index[:10]       # kullanıcının gördüğü ilk 10
            rows.append({
                "query": item["id"], "yontem": "vektor", "text": text,
                "P@10": M.precision_at_k(s, rel, 10), "R@10": M.recall_at_k(s, rel, 10),
                "R@20": M.recall_at_k(s, rel, 20), "AUC": M.auc(s, rel),
                "AUC_sekans_ici": w_auc, "sekans_sayisi": n_seq,
                "R@20_buyuk": M.subgroup_recall_at_k(s, rel, group, "big", 20),
                "R@20_kucuk": M.subgroup_recall_at_k(s, rel, group, "small", 20),
                "farkli_klip_ilk10": lab.loc[top10, "clip"].nunique(),
            })
    rows += evaluate_hybrid(engine, queries, lab)
    return pd.DataFrame(rows)


def evaluate_hybrid(engine: SearchEngine, queries: dict, lab: pd.DataFrame) -> list[dict]:
    """Filtreli sorgular: P@k ve R@k tüm havuzda, AUC yalnızca filtre içinde (karar günlüğü, 18 Kas)."""
    bases = {q["id"]: q for q in queries["queries"]}
    rows = []
    for h in queries.get("hybrid", []):
        base = bases[h["base"]]
        if base["target"]["type"] != "object":
            raise ValueError("Hibrit sorgu yalnızca nesne sorgusuna bağlanabilir")

        # F: filtreden geçen segmentler (aramayla aynı kod); doğru = nesne var VE F, F dışı yanlış
        F = pd.Series(engine.mask(h["filter"]), index=lab.index)
        rel = relevance(lab, base["target"])["rel"].where(F, 0.0)
        relF = rel[F]
        info = {"query": base["id"], "filtre": h["id"],
                "n_dogru_F": int((relF == 1).sum()), "n_yanlis_F": int((relF == 0).sum())}

        # İfadeden bağımsız tabanlar: tüm havuzda rastgele ve yalnızca F içinde rastgele (formülle)
        for yontem, text, r in [("rastgele", "(rastgele taban)", rel),
                                ("filtre_rastgele", "(filtre + rastgele)", relF)]:
            b10, b20 = M.random_baseline(r, 10), M.random_baseline(r, 20)
            rows.append({**info, "yontem": yontem, "text": text, "P@10": b10["precision"],
                         "R@10": b10["recall"], "R@20": b20["recall"], "AUC_filtre_ici": 0.5})

        # Her ifade: yalnız vektör tüm havuzda, hibrit yalnızca F içinde (doğruların hepsi F'de, sonuç aynı)
        for text in base["texts"]:
            s = pd.Series(engine.scores(text), index=lab.index)
            auc_f = M.auc(s[F], relF)                               # iki satırda aynı: skorlar aynı
            for yontem, ss, rr in [("vektor", s, rel), ("hibrit", s[F], relF)]:
                top10 = ss.sort_values(ascending=False).index[:10]
                rows.append({**info, "yontem": yontem, "text": text,
                             "P@10": M.precision_at_k(ss, rr, 10), "R@10": M.recall_at_k(ss, rr, 10),
                             "R@20": M.recall_at_k(ss, rr, 20), "AUC_filtre_ici": auc_f,
                             "farkli_klip_ilk10": lab.loc[top10, "clip"].nunique()})
    return rows


def main():
    p = argparse.ArgumentParser(description="Puan tablosuna satır ekle")
    p.add_argument("--label", required=True, help="deney adı, ör. v1")
    p.add_argument("--index", default=DEFAULT_INDEX, help="indeks klasör adı")
    p.add_argument("--dry-run", action="store_true", help="tabloya yazmadan yalnızca ekrana bas")
    args = p.parse_args()

    queries = load_queries()
    engine = SearchEngine(args.index)
    res = evaluate(engine, queries, segment_labels())

    # Satırın hangi koşulda üretildiği: deney, indeks, sorgu dosyası sürümü, tarih
    res.insert(0, "label", args.label)
    res.insert(1, "index", args.index)
    res.insert(2, "queries_version", queries["version"])
    res["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 45)
    plain = res[res["filtre"].isna()]
    hyb = res[res["filtre"].notna()]
    print(plain[["query", "yontem", "text", "P@10", "R@10", "R@20", "AUC", "AUC_sekans_ici",
                 "R@20_buyuk", "R@20_kucuk", "farkli_klip_ilk10"]].round(3).to_string(index=False))
    print()
    print(hyb[["filtre", "yontem", "text", "n_dogru_F", "n_yanlis_F", "P@10", "R@10", "R@20",
               "AUC_filtre_ici", "farkli_klip_ilk10"]].round(3).to_string(index=False))

    if args.dry_run:
        print(f"\ndry-run: tabloya yazilmadi ({len(res)} satir)")
        return

    # Sütunlar sürümler arasında değişebilir: dosyanın sonuna eklemek sütunları kaydırırdı.
    # Eski tabloyla birleştirip baştan yaz; geçici dosya + replace ile yarıda kesilirse eski tablo bozulmaz.
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SCORES_PATH.exists():
        res = pd.concat([pd.read_csv(SCORES_PATH, encoding="utf-8"), res], ignore_index=True)
    tmp = SCORES_PATH.with_suffix(".tmp")
    res.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(SCORES_PATH)
    print(f"\nyazildi: {SCORES_PATH} (toplam {len(res)} satir)")


if __name__ == "__main__":
    main()