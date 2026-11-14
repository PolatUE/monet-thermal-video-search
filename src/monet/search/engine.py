"""Metin sorgusu -> en benzer segmentler. Arayüzün backend'i bu sınıfı kullanır."""
import numpy as np
import pandas as pd

from monet.config import OUTPUT_DIR
from monet.indexing.build import load_index
from monet.indexing.frames import FPS
from monet.models.clip import ClipModel

DEFAULT_INDEX = "ViT-B-32__laion2b_s34b_b79k"
RESULT_COLUMNS = ["rank", "score", "clip", "seg", "video_bas", "video_son", "t_bas", "t_son", "seq", "n_frames"]


class SearchEngine:
    """Açılışta indeks ve model bir kez yüklenir; search() her sorguda yalnızca metni kodlar ve karşılaştırır."""

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

    def search(self, query: str, k: int = 10, filters: dict | None = None) -> pd.DataFrame:
        """En benzer k segmenti skor sırasıyla döndürür. filters v1'de (telemetri filtresi) eklenecek."""
        if filters:
            raise NotImplementedError("Telemetri filtresi v1'de eklenecek")
        q = self.model.encode_text([query])[0]          # (512,) birim vektör
        scores = self.vectors @ q                       # birim vektörlerde iç çarpım = kosinüs benzerliği
        top = np.argsort(-scores)[:k]                   # büyükten küçüğe ilk k
        res = self.meta.iloc[top].copy()
        res["rank"] = np.arange(1, len(top) + 1)
        res["score"] = scores[top].round(4)
        return res[RESULT_COLUMNS].reset_index(drop=True)