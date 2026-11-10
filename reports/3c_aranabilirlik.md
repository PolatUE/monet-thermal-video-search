# 3c. Aranabilirlik — sonuç raporu

**Sonuç: SORUNLU (K3, person).** Sorun veride değil, mevcut arama planında: insanlar 2×2 tiling ile görülemeyecek kadar küçük. Veri, küçük nesne problemini ölçmek için iyi bir test zemini.

Üretim: `scripts/02_build_frame_table.py` → `outputs/annotations.parquet`,
`scripts/10_segments.py` → `outputs/segments.parquet`,
`scripts/11_object_size_contrast.py` → `outputs/suitability/box_stats.parquet`

## Segment tanımı

14 video klibi 5 sn'lik segmentlere bölündü; klibin sonundaki 5 sn'den kısa parça dahil edilmedi. **383 segment** (telemetrili 4 sekanstan 265).

| Sekans | Segment | İrtifa medyanı | Açı medyanı | Güneş yüksekliği | Person var | Vehicle var |
|---|---|---|---|---|---|---|
| dirtroad_01 | 74 | — | — | 3,2…4,4° | 74 | 74 |
| dirtroad_04 | 69 | 100 m | −30° | 17,2…17,9° | 51 | 47 |
| dirtroad_05 | 91 | 99 m | −25° | 14,9…16,1° | 85 | 2 |
| dirtroad_06 | 43 | 100 m | −35° | 11,6…12,2° | 43 | 1 |
| dirtroad_07 | 62 | 105 m | −30° | 8,9…9,8° | 62 | 62 |
| runway_01 | 17 | — | — | 4,5…4,8° | 13 | 0 |
| runway_02 | 27 | — | — | −9,7…−8,5° | 19 | 11 |

Güneş yüksekliği, segment ortasındaki gerçek zaman (`t_src`) ve konumdan astronomik formülle hesaplandı (telemetrisiz sekanslarda MONET bölgesinin yaklaşık konumu). Doğrulama: 15 Aralık öğle 31,6°, 13 Aralık gün batımı ~16:35.

## Eşikler (onaylı)

| Kontrol | Uygun |
|---|---|
| K1 Sahne çeşitliliği | Her boyutta en az iki bandın her biri segmentlerin ≥ %10'u |
| K2 Nesne var/yok | Her sınıf için var ve yok segmentleri ≥ 20 ve ≥ %5 (var = segmentin en az bir gerçek karesinde o sınıftan kutu) |
| K3 Nesne boyutu | Kutuların ≥ %80'i 2×2 tiling'de en az 1 yama (14 px) büyüklüğünde; model girişi 224 px varsayımı |
| K4 Kenar | Yalnızca rapor (ortadan kare kırpmada kaybolan sol/sağ 100 px) |
| K5 Kontrast | Kutu içi ile 5 px çevre halkası arasındaki parlaklık farkı < 10 gri seviye olan kutuların oranı ≤ %10 (her 5. kare örneklendi) |

## K1: Sahne çeşitliliği — Uygun, uç bantlar sınırlı

| Boyut | Bant | Segment | Pay |
|---|---|---|---|
| İrtifa | yüksek ≥ 100 m | 163 | %61,5 |
| | orta 60–100 m | 100 | %37,7 |
| | alçak < 60 m | 2 | %0,8 |
| Kamera açısı | orta −45…−30° | 133 | %50,2 |
| | sığ > −30° | 126 | %47,5 |
| | dik < −45° | 6 | %2,3 |
| Güneş | yüksek > 10° | 203 | %53,0 |
| | alçak 0–10° | 153 | %39,9 |
| | batım sonrası < 0° | 27 | %7,0 |

- Pratikte her boyutta iki bant test edilebilir. Alçak irtifa, dik açı ve gece sorguları bu veriyle güvenilir biçimde test edilemez (kapsam sınırı).
- Güneş bandı sekansla birebir örtüşüyor (her sekans tek zaman diliminde).
- Bazı sekansların medyanı bant sınırında (100 m, −30°); sınıra yakın segmentler etiket açısından kırılgan.

## K2: Nesne var/yok — Uygun, sekans etkisi var

| Sınıf | Var | Yok |
|---|---|---|
| person | 347 (%90,6) | 36 (%9,4) |
| vehicle | 197 (%51,4) | 186 (%48,6) |

- Person "yok" segmentleri 4 sekanstan: dirtroad_04 (18), runway_02 (8), dirtroad_05 (6), runway_01 (4).
- Vehicle varlığı sekansla örtüşüyor: dirtroad_01 ve 07'de her segmentte var; dirtroad_05, 06 ve runway_01'de neredeyse hiç yok. Hem var hem yok segmenti olan yalnızca dirtroad_04 (47/22) ve runway_02 (11/16). Bir model aracı görmeden sahneyi tanıyarak yüksek puan alabilir.

## K3: Nesne boyutu — SORUNLU (person)

Boyut = ortalama kenar √(w × h). Görünürlük sınırı (224 px giriş, 14 px yama): 1×1 → 37,5 px, 2×2 → 18,75 px, 3×3 → 12,5 px, 4×4 → 9,4 px.

| | Person | Vehicle |
|---|---|---|
| Kutu | 58 176 | 11 838 |
| Medyan boyut | 14,1 px | 38,3 px |
| Görünür, 1×1 | %5,3 | %52,0 |
| Görünür, 2×2 | **%12,9** | %99,1 |
| Görünür, 3×3 | %70,6 | %99,9 |
| Görünür, 4×4 | %96,8 | %100 |

- Global (tüm kare) embedding'de insanların ~%95'i bir yamadan küçük.
- MVP'deki 2×2 tiling ile yeniden sıralama insan sorguları için yetersiz; ≥ %80 için 4×4 gerekiyor.
- 336 px girişli bir modelde 2×2 sınırı 12,5 px olur (~%71 görünür), 3×3 sınırı ~8,3 px (~%97).

**Faz 3/5'te değerlendirilecek seçenekler:** 4×4 tiling (2×2'ye göre 4 kat CPU işi), daha yüksek çözünürlüklü model, küçük nesne dedektörüyle tamamlama (sabit sınıflı; açık uçlu sorgulara ölçeklenmez). Önerilen ölçüt: hangi model + tiling birleşimi küçük insanları bulabiliyor.

## K4: Kenardaki nesneler — Rapor

Ortadan kare kırpmada tamamen kaybolan kutular: person %9,6, vehicle %6,0. Faz 3'te kırpma yerine kenar ekleme (padding) tercih edilmeli.

## K5: Kontrast — Uygun, belirli sekanslarda yoğun

Ölçülen kutu: person 11 646, vehicle 2369. Medyan fark: person 53,3, vehicle 41,8 gri seviye.
Düşük kontrast oranı: person %4,3, vehicle %6,5.

| Sekans | Güneş | Person | Vehicle |
|---|---|---|---|
| dirtroad_04 / 05 / 06 | 11–18° | %0,8 / %1,1 / %1,4 | %0,2 / %0 / %0 |
| dirtroad_07 | 9–10° | %1,2 | %5,8 |
| dirtroad_01 | 3–4° | **%24,9** | %6,7 |
| runway_01 | 4–5° | **%10,5** | — |
| runway_02 | −9° | %0 | **%30,1** |

- Yüksek güneşte düşük kontrast yok denecek kadar az. Gün batımına yakın sekanslarda insanlar, batım sonrasında araçlar zeminle karışıyor (runway_01_p2 ~1:09'da gözle de görüldü).
- Olası açıklama (doğrulanmadı): ısınmış zemin ile insan vücudunun benzer sıcaklıkta olması; gece duran araçların ortam sıcaklığına soğuması. dirtroad_07 alçak güneşte olduğu halde düşük kaldığı için güneş yüksekliği tek açıklama olmayabilir.

## 3d'ye devredilenler

- Sekans içi (nesneyi görme) ve sekanslar arası (gerçek kullanım) ölçümünü ayırmak.
- Bant sınırına yakın segmentleri doğru cevap tanımında dışarıda bırakmak.
- Düşük kontrastlı ve çok küçük kutuları (`box_stats.parquet`) doğru cevaplarda ayrı tutmak.
- Test edilemeyen sorgu türlerini (alçak irtifa, dik açı, gece, hava durumu) kapsam dışı olarak yazmak.