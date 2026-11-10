# 3a. Zaman tutarlılığı — sonuç raporu

**Sonuç: SORUNLU (7 sekansın 5'inde).** Sorunların nedeni belirlendi; hepsi için bir işleme yolu var.

Üretim: `scripts/02_build_frame_table.py` → `outputs/frames.parquet`,
`scripts/03_time_check.py` → `outputs/suitability/time_check.csv`,
`scripts/04_clock_offset.py` → `outputs/suitability/clock_offset.png`

## Eşikler (onaylı)

| Kontrol | Uygun |
|---|---|
| K1 Zaman geri gitmesi | 0 kez |
| K2 fps | Standart değere (25 / 29,97 / 30) ≤ %2 uzak; değilse belirsiz |
| K3 Üç parçadaki fps farkı | ≤ %2 |
| K4 Artık (25 fps regresyonuna göre) | Karelerin ≥ %99'unda ≤ 1 sn, art arda > 2 sn yok |
| K5 Boşluk | En uzun boşluk ≤ 5 sn ve 5 sn'lik segmentlerin ≥ %95'inde ≥ 8 gerçek kare |

R² yalnızca bilgi amaçlı; karar vermiyor (dirtroad_05: R² = 0,99975, ama artık 5,5 sn'ye çıkıyor).

## Sekans bazında sonuç

| Sekans | fps (reg.) | Parça fps | Maks. artık | Maks. boşluk | Seg. ≥ 8 kare | Sonuç | Neden |
|---|---|---|---|---|---|---|---|
| dirtroad_01 | 24,99 | 24,96 / 24,99 / 24,96 | 0,65 sn | 24,1 sn | %90,5 | Sorunlu (K5) | Veri boşluğu |
| dirtroad_04 | 24,97 | 25,01 / 24,94 / 24,99 | 1,29 sn | 1,9 sn | %100 | Uygun | — |
| dirtroad_05 | 24,36 | 22,70 / 24,99 / 24,54 | 5,53 sn | 1,9 sn | %100 | Sorunlu (K2 belirsiz, K3, K4) | Kayıt takılması |
| dirtroad_06 | 24,98 | 24,97 / 24,98 / 24,95 | 0,57 sn | 1,2 sn | %100 | Uygun | — |
| dirtroad_07 | 24,81 | 25,04 / 24,97 / 24,11 | 1,91 sn | 1,4 sn | %100 | Sorunlu (K3, K4) | Kayıt takılması |
| runway_01 | 24,98 | 24,92 / 24,99 / 24,95 | 0,59 sn | 20,2 sn | %81,0 | Sorunlu (K5) | Veri boşluğu |
| runway_02 | 24,97 | 24,97 / 24,97 / 24,68 | 0,61 sn | 153,5 sn | %44,2 | Sorunlu (K5) | Veri boşluğu |

## Bulgular

1. **Kamera 25 fps, kaydedilen her 2. kare.** Yedi sekansta da kare numaraları tek ve tekrarsız. Etkin örnekleme 12,5 kare/sn.
2. **fps kare numarasından hesaplanır, dosya sayısından değil.** Örnek dirtroad_04: 8732 / 348 ≈ 25; 3195 / 348 ≈ 9,2 yalnızca elimizdeki dosya yoğunluğu.
3. **Veri boşluğu (dirtroad_01, runway_01, runway_02).** Saat ile kare numarası uyumlu (K1–K4 uygun), ama bazı bölümlerde hiç kare yok. runway_02'de 389 sn'nin 153,5 sn'si boş.
4. **Kayıt takılması (dirtroad_05, dirtroad_07).** Kayıttaki zaman ile 25 fps'e göre beklenen zaman arasındaki fark basamaklar halinde artıyor:
   - dirtroad_05: ~70., ~100., ~135. ve ~325. video saniyelerinde; toplam +13,7 sn.
   - dirtroad_07: ~247. saniyede; +2,7 sn.
   - Basamaklar tek bir karede değil, birkaç saniyelik video boyunca yükseliyor. Bu şekil saat atlamasından çok kaydın yavaşlamasına/takılmasına uyuyor. Ön sonuç: `date_captured` gerçek zamanı, kare numarası kaydedilen kare sırasını gösteriyor.
   - Ardışık iki kareye bakan sıçrama aramayla bu basamaklar bulunamadı; tespit grafikle yapıldı.
5. **Görüntü–telemetri eşleşmesi etkilenmiyor.** MONET'te telemetri her karenin kendi kaydında. Etkilenen, video saniyesi ile gerçek saat arasındaki eşleme.

## Kararlar

- Zaman ekseni kare numarasından kurulur; her karenin gerçek zamanı (`t`) tabloda ayrıca saklanır. Arama sonucu hem video saniyesini hem gerçek saati döndürür. (Seçenek A)
- Kontrol 5'in eşiği, toplam eksik oranı yerine segment başına gerçek kare sayısına göre yeniden tanımlandı.

## Açık konular

- Video kare hızı: 12,5 fps ızgarası öneriliyor (bütün sekanslara uyuyor), onay bekliyor.
- "Boşlukta böl" kuralı (5 sn'den uzun boşlukta sekansı ayrı videolara bölmek): ileride değerlendirilecek.
- Kayıt takılması yorumunun telemetriyle doğrulanması: 3b'de yapılacak (basamak öncesi/sonrası GPS konum farkı).