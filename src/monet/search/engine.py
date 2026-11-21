"""Metin sorgusu -> en benzer segmentler. Arayüzün backend'i bu sınıfı kullanır."""
import numpy as np
import pandas as pd

from monet.config import OUTPUT_DIR
from monet.indexing.build import load_index
from monet.indexing.frames import FPS
from monet.models.clip import ClipModel
from monet.search.filters import filter_keys

DEFAULT_INDEX = "ViT-B-32__laion2b_s34b_b79k"
RESULT_COLUMNS = ["rank", "score", "clip", "seg", "video_bas", "video_son", "t_bas", "t_son", "seq", "n_frames"]


class SearchEngine:
    """Açılışta indeks ve model bir kez yüklenir; her sorguda yalnızca metin kodlanır ve karşılaştırılır."""

    def __init__(self, index_name: str = DEFAULT_INDEX):
        self.index_name = index_name
        self.table, self.vectors, self.info = load_index(index_name)
        # Sorgu, indeksi üreten modelle kodlanmalı (aynı vektör uzayı); model bilgisi index_info.json'dan
        self.model = ClipModel(self.info["arch"], self.info["pretrained"])

        # Segment bilgileri indekste tutulmuyor; clip + seg ile segments.parquet'ten eşleştir
        seg = pd.read_parquet(OUTPUT_DIR / "segments.parquet")
        cols = ["clip", "seg", "seq", "video_bas", "yuva", "t_bas", "t_son"]
        self.meta = self.table.merge(seg[cols], on=["clip", "seg"], how="left", validate="one_to_one")
        assert len(self.meta) == len(self.vectors)      # meta'nın i. satırı = vektörlerin i. satırı
        self.meta["video_son"] = self.meta["video_bas"] + self.meta["yuva"] / FPS

    def scores(self, query: str) -> np.ndarray:
        """Sorgunun tüm segmentlerle kosinüs benzerliği; i. değer = meta'nın i. satırı."""
        q = self.model.encode_text([query])[0]          # (512,) birim vektör
        return self.vectors @ q                         # birim vektörlerde iç çarpım = kosinüs benzerliği

    def mask(self, filters: dict) -> np.ndarray:
        """Filtreden geçen segmentler True; i. değer = meta'nın i. satırı (değerlendirme de bunu kullanır)."""
        allowed = pd.MultiIndex.from_frame(filter_keys(filters))
        return pd.MultiIndex.from_frame(self.meta[["clip", "seg"]]).isin(allowed)

    def search(self, query: str, k: int = 10, filters: dict | None = None) -> pd.DataFrame:
        """En benzer k segmenti skor sırasıyla döndürür. filters varsa önce filtre (ön filtre), sonra sıralama."""
        scores = self.scores(query)
        if filters:
            allowed = self.mask(filters)
            scores = np.where(allowed, scores, -np.inf)  # filtre dışı en sona düşer
            k = min(k, int(allowed.sum()))              # filtreden k'dan az geçerse daha az sonuç
        top = np.argsort(-scores)[:k]                   # büyükten küçüğe ilk k
        res = self.meta.iloc[top].copy()
        res["rank"] = np.arange(1, len(top) + 1)
        res["score"] = scores[top].round(4)
        return res[RESULT_COLUMNS].reset_index(drop=True)