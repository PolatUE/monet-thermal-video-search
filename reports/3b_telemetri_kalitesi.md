# 3b. Telemetri kalitesi — sonuç raporu

**Sonuç: UYGUN (telemetrili 4 sekansın 4'ü).** dirtroad_01, runway_01, runway_02 telemetrisiz (12 alan boş metin); bu sekanslar yalnızca görüntüyle aranır.

Üretim: `scripts/08_telemetry_check.py` → `outputs/suitability/telemetry_check.csv`,
`scripts/09_stall_check.py` → `outputs/suitability/stall_check.csv`

## Eşikler (onaylı)

| Kontrol | Uygun |
|---|---|
| K1 Doluluk | ≥ %95 (%0 = telemetrisiz) |
| K2 Güncelleme | ≥ 1 Hz ve en uzun donma ≤ 2 sn (12 alandan herhangi biri değişirse güncelleme) |
| K3 Fiziksel aralık | Her alanda aralık dışı ≤ %1 |
| K4 GPS sıçraması | 1 sn'lik aralıkların ≥ %99'unda GPS hızı ≤ 30 m/sn |
| K5 Hız birimi | 5 sn'lik pencerelerde (ortalama hız alanı / GPS hızı) dağınıklığı ≤ 0,25 |
| K6 Kayıt takılması | Basamakta GPS'ten ima edilen süre saate mi kare numarasına mı yakın |

## Sekans bazında sonuç

| Sekans | Güncelleme | Maks. donma | GPS hızı medyan / maks. | Hız oranı | Dağınıklık | Sonuç |
|---|---|---|---|---|---|---|
| dirtroad_04 | 3,99 Hz | 1,04 sn | 2,81 / 8,8 m/sn | 101,3 | 0,110 | Uygun |
| dirtroad_05 | 4,63 Hz | 0,72 sn | 3,16 / 17,6 m/sn | 101,8 | 0,108 | Uygun |
| dirtroad_06 | 4,87 Hz | 0,56 sn | 3,69 / 5,5 m/sn | 102,3 | 0,105 | Uygun |
| dirtroad_07 | 4,52 Hz | 0,80 sn | 3,06 / 6,1 m/sn | 102,1 | 0,101 | Uygun |

## Alanların anlamı (ölçümle belirlendi)

- `speed_x`: yatay hız, **cm/sn** (5 sn'lik pencerelerde GPS hızının ~101 katı).
- `speed_y`, `gimbal_roll`: her yerde 0, bilgi taşımıyor.
- `gimbal_yaw`: kameranın **drone'a göre** yatay açısı (0 = ileri); −33…+52°.
- `UAV_yaw`: drone'un pusula yönü, 0–360°.
- `altitude`: 43–120 m; dört sekansta da üst sınır tam 120 m.
- `gimbal_pitch`: −50…+19°; dirtroad_04 dışında hep −20° ve aşağısı.

## K6: Kayıt takılması doğrulaması

| Sekans | Video sn | Kare süresi | Saat süresi | GPS yolu | İma edilen süre | Sonuç |
|---|---|---|---|---|---|---|
| dirtroad_05 | 70,5 | 10,0 | 18 | 49,8 m | 18,2 | (ii) saat doğru |
| dirtroad_05 | 98,9 | 10,0 | 12 | 30,1 m | 11,5 | (ii) saat doğru |
| dirtroad_05 | 136,6 | 10,0 | 12 | 35,1 m | 12,7 | (ii) saat doğru |
| dirtroad_05 | 322,2 | 10,0 | 14 | 38,3 m | 14,4 | (ii) saat doğru |
| dirtroad_07 | 244,2 | 10,0 | 13 | 34,0 m | 13,0 | (ii) saat doğru |

Yöntem kontrolü (basamaksız yerler, ima / gerçek süre medyanı): 1,001 / 1,021 / 1,011 / 1,025.

**Sonuç:** `date_captured` gerçek zamanı gösteriyor; takılma anlarında kayıt sistemi az kare kaydedip bunları ardışık numaralandırmış. Seçenek A doğrulandı: video zamanı kare numarasından, gerçek zaman `t_src`'den alınır.

## Yöntemde yapılan düzeltmeler

- `gimbal_yaw` kabul aralığı 0–360° yerine −180…+180° (alan drone'a göre açı).
- K5: 1 sn'lik anlık karşılaştırma GPS gürültüsü nedeniyle dağınıktı (0,52–0,86). 5 sn pencere + pencere ortalaması ile 0,10'a indi. Eşik değişmedi.