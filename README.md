# MONET termal drone kayıtlarında metinle segment arama

Açık bir termal drone veri seti (MONET) üzerinde, metin sorgusu ve uçuş telemetrisiyle ilgili video segmentini bulan bir arama sistemi için kişisel çalışma.

Hedef akış: kareler → zaman ekseni doğru videolar → 5 sn'lik segmentler → embedding + telemetri filtresi (hibrit arama).

**Durum:** Veri seti uygunluk analizi (3a zaman tutarlılığı, 3b telemetri kalitesi, 3c aranabilirlik) tamamlandı; 3d (değerlendirme seti) devam ediyor.

## Öne çıkan bulgular

- Kamera 25 fps; veri setinde her 2. kare var. Kareler 12,5 fps videolara çevrildi; 5 sn'den uzun boşluklarda kayıt ayrı kliplere bölündü → **14 klip, 32,6 dk**.
- İki sekansta kayıt takılması bulundu (saat, kare numarasından 13,7 sn'ye kadar önde). GPS yolu / hız ile doğrulandı: `date_captured` gerçek zamanı gösteriyor. Video zamanı kare numarasından, gerçek zaman ayrı sütundan tutuluyor.
- `speed_x` alanı cm/sn cinsinden yatay hız (5 sn'lik pencerelerde GPS hızının ~101 katı, dağınıklık ~0,10).
- 383 segment. İnsan kutularının medyan boyutu ~14 px: 224 px girişli bir modelde 2×2 tiling ile yalnızca ~%13'ü bir yama (14 px) büyüklüğüne ulaşıyor; ≥ %80 için 4×4 gerekiyor.
- Düşük termal kontrast gün batımına yakın ve sonrasındaki sekanslarda yoğunlaşıyor.

Ayrıntılar: [`reports/`](reports/)

## Kurulum (Windows, Python 3.13)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

Ham veri yolu koda yazılmaz; `MONET_RAW_DIR` ortam değişkeninden okunur:

```powershell
setx MONET_RAW_DIR "C:\yol\monet_dataset"   # sonra terminali yeniden aç
```

Beklenen klasör yapısı (her sekans için): `<sekans>/annotations/labels.json` ve `<sekans>/lwir/imageN.png`.
Ham veri salt okunur kullanılır; tüm çıktılar `outputs/` altına yazılır (repoya dahil değil).

## Script'ler (çalışma sırası)

| Script | Ne yapar | Çıktı |
|---|---|---|
| `01_inspect_labels.py` | `labels.json` yapısını ve dosya eşleşmesini özetler | ekran |
| `02_build_frame_table.py` | Kare ve kutu tabloları | `frames.parquet`, `annotations.parquet` |
| `03_time_check.py` | 3a: geri gitme, fps, üç parça, artıklar, boşluklar | `suitability/time_check.csv` |
| `04_clock_offset.py` | Saat − kare zamanı farkı grafiği | `suitability/clock_offset.png` |
| `05_inspect_png.py` | PNG formatı (8 bit RGB, 800×600) | ekran |
| `06_plan_clips.py` | Boşlukta bölme, 12,5 fps yuva planı | `clips.parquet`, `slots.parquet` |
| `07_encode_videos.py` | H.264 MP4 üretimi (PyAV) | `videos/*.mp4` |
| `08_telemetry_check.py` | 3b: doluluk, güncelleme, aralık, GPS, hız birimi | `suitability/telemetry_check.csv` |
| `09_stall_check.py` | 3b: kayıt takılmasının GPS ile doğrulanması | `suitability/stall_check.csv` |
| `10_segments.py` | 5 sn'lik segment tablosu; 3c sahne çeşitliliği ve nesne var/yok | `segments.parquet` |
| `11_object_size_contrast.py` | 3c: nesne boyutu / tiling, kenar, kontrast | `suitability/box_stats.parquet` |

## Veri seti ve atıf

Bu repo veri setini içermez. MONET, CC BY 4.0 lisansıyla yayımlanmıştır.

L. Riz, A. Caraffa, M. Bortolon, M. L. Mekhalfi, D. Boscaini, A. Moura, J. Antunes, A. Dias, H. Silva, A. Leonidou, C. Constantinides, C. Keleshis, D. Abate, F. Poiesi. *The MONET Dataset: Multimodal Drone Thermal Dataset Recorded in Rural Scenarios.* CVPR Workshops, 2023, s. 2546–2554. [arXiv:2304.05417](https://arxiv.org/abs/2304.05417)

```bibtex
@InProceedings{Riz_2023_CVPR,
    author    = {Riz, Luigi and Caraffa, Andrea and Bortolon, Matteo and Mekhalfi, Mohamed Lamine and Boscaini, Davide and Moura, Andr\'e and Antunes, Jos\'e and Dias, Andr\'e and Silva, Hugo and Leonidou, Andreas and Constantinides, Christos and Keleshis, Christos and Abate, Dante and Poiesi, Fabio},
    title     = {The MONET Dataset: Multimodal Drone Thermal Dataset Recorded in Rural Scenarios},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    year      = {2023},
    pages     = {2546-2554}
}
```