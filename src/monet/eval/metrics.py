"""Sıralama ölçüleri: bir sorgunun segment skorları + doğru cevaplar (1 / 0 / NaN) -> sayılar."""
import numpy as np
import pandas as pd


def _ranked(scores: pd.Series, rel: pd.Series) -> np.ndarray:
    """Değerlendirme dışını (NaN) çıkarır, kalanların 1/0 değerini skor sırasıyla (büyükten küçüğe) döndürür."""
    m = rel.notna()
    order = np.argsort(-scores[m].to_numpy(), kind="stable")
    return rel[m].to_numpy()[order]


def precision_at_k(scores: pd.Series, rel: pd.Series, k: int) -> float:
    r = _ranked(scores, rel)[:k]
    return float(r.mean()) if len(r) else np.nan


def recall_at_k(scores: pd.Series, rel: pd.Series, k: int) -> float:
    r = _ranked(scores, rel)
    n_pos = r.sum()
    return float(r[:k].sum() / n_pos) if n_pos else np.nan


def auc(scores: pd.Series, rel: pd.Series) -> float:
    """Doğru bir segmentin yanlış bir segmentten yüksek skor alma olasılığı (sıra toplamı formülü)."""
    m = rel.notna()
    s, r = scores[m], rel[m]
    n_pos, n_neg = int((r == 1).sum()), int((r == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = s.rank(method="average")                      # küçükten büyüğe sıra; eşit skorlar ortalama sıra alır
    return float((ranks[r == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def within_group_auc(scores: pd.Series, rel: pd.Series, groups: pd.Series) -> tuple[float, int]:
    """Her grubun (sekansın) kendi içindeki AUC'lerinin ortalaması ve kaç grupta hesaplanabildiği."""
    vals = [auc(scores[groups == g], rel[groups == g]) for g in groups.unique()]
    vals = [v for v in vals if not np.isnan(v)]           # tek sınıflı sekanslar hesaba girmez
    return (float(np.mean(vals)) if vals else np.nan), len(vals)


def subgroup_recall_at_k(scores: pd.Series, rel: pd.Series, group: pd.Series, name: str, k: int) -> float:
    """Alt gruptaki doğru segmentlerin ne kadarı tüm sıralamanın ilk k'sında."""
    m = rel.notna()
    order = scores[m].sort_values(ascending=False, kind="stable").index
    top = set(order[:k])
    members = group[m][(rel[m] == 1) & (group[m] == name)].index
    return float(sum(i in top for i in members) / len(members)) if len(members) else np.nan


def random_baseline(rel: pd.Series, k: int) -> dict:
    """Rastgele sıralamanın beklenen değerleri (hesapla; rastgele sayı gerekmez)."""
    n = int(rel.notna().sum())
    n_pos = int((rel == 1).sum())
    return {"precision": n_pos / n, "recall": min(k, n) / n, "auc": 0.5}