"""OpenCLIP görüntü-metin modeli için sarmalayıcı: bir kez yükle, görüntü ve metni aynı uzayda vektöre çevir."""
import numpy as np
import open_clip
import torch
from PIL import Image


class ClipModel:
    """Model bir kez yüklenir; encode_images ve encode_text birim uzunlukta (L2 normalize) vektör döndürür."""

    def __init__(self, arch: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str = "cpu"):
        self.arch, self.pretrained, self.device = arch, pretrained, device
        # preprocess: modelin kendi ön işlemesi (224'e küçültme, normalizasyon)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model.eval().to(device)                     # eğitim değil çıkarım modu (dropout vb. kapalı)
        self.tokenizer = open_clip.get_tokenizer(arch)   # metni modelin anladığı token dizisine çevirir

    @property
    def name(self) -> str:
        """İndeks klasör adı olarak kullanılır: outputs/index/<name>/"""
        return f"{self.arch}__{self.pretrained}"

    @torch.inference_mode()                              # gradyan tutma: daha az bellek, daha hızlı
    def encode_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Görüntüleri parti parti vektöre çevirir. Dönüş: (N, D) float32, her satır birim uzunlukta."""
        out = []
        for i in range(0, len(images), batch_size):
            x = torch.stack([self.preprocess(im) for im in images[i:i + batch_size]]).to(self.device)
            out.append(self._normalize(self.model.encode_image(x)))
        return np.concatenate(out) if out else np.empty((0, self.dim), dtype=np.float32)

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Metinleri vektöre çevirir. Dönüş: (N, D) float32, her satır birim uzunlukta."""
        tokens = self.tokenizer(texts).to(self.device)
        return self._normalize(self.model.encode_text(tokens))

    @property
    def dim(self) -> int:
        """Vektör uzunluğu (ViT-B/32 için 512)."""
        return self.model.text_projection.shape[1]

    @staticmethod
    def _normalize(v: torch.Tensor) -> np.ndarray:
        # Birim uzunluk: kosinüs benzerliği = iç çarpım olur
        v = v / v.norm(dim=-1, keepdim=True)
        return v.float().cpu().numpy()