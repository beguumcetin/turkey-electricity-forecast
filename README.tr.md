[English](README.md) | Türkçe

# Türkiye Ertesi Gün Elektrik Talep Tahmini

Türkiye'nin ulusal elektrik tüketimi için saatlik, ertesi güne dönük bir tahmin; sonuçlar EPİAŞ'ın (Enerji Piyasaları İşletme A.Ş.) yayımladığı resmî ertesi gün yük planıyla karşılaştırılıyor. Soru şu: takvim, tatil, hava durumu ve yakın geçmiş tüketim özelliklerinden kurulan bir model, şebeke işletmecisinin zaten yayımladığı planı geçebilir mi?

Ayrılmış bir yılda LightGBM modeli yük planını küçük bir farkla geçiyor (ortalama mutlak yüzde hata %2,27'ye karşı %2,60), ama karşılaştırma modelin biraz lehine; bkz. [Karşılaştırma ne kadar adil](#karşılaştırma-ne-kadar-adil).

## Tasarım

- **Ertesi gün senaryosu.** D günü için tahmin, D-1'in sabahında yapılır; o anda tamamlanmış son tüketim günü D-2'dir. Bu yüzden her gecikme (lag) özelliği en az 48 saatlik; model pratikte elinde olmayacak bir bilgiyi görmez.
- **Dürüst değerlendirme.** Son 12 ay test dönemi olarak ayrılır, öncesi eğitimde kullanılır. Test döneminden hiçbir şey bir modeli eğitmek ya da ayarlamak için kullanılmaz.
- **Geçilmesi gereken bir referans.** EPİAŞ'ın kendi ertesi gün yük planı, modellerle aynı saatlerde puanlanır.

## Özellikler

- **Takvim:** saat, haftanın günü, ay, yılın gününün sinüs/kosinüs çifti, hafta sonu işareti.
- **Tatiller** (Türkiye için `holidays` paketinden): resmî tatiller, bayramlar, saat 13:00'te başlayan yarım gün tatiller, köprü günleri (bir tatil ile hafta sonu arasında kalan iş günü) ve en yakın bayrama gün cinsinden uzaklık (en çok 7); çünkü uzun bayramların çevresinde tüketim birkaç gün düşük kalıyor.
- **Hava durumu:** nüfusa göre en büyük altı il (İstanbul, Ankara, İzmir, Bursa, Antalya, Adana) için Open-Meteo arşivinden saatlik sıcaklık; nüfus ağırlıklarıyla tek bir ulusal seride birleştirilir. Buradan ısıtma ve soğutma dereceleri (18 °C ve 22 °C tabanlı) ile 24 saatlik ortalama türetilir.
- **Yakın geçmiş tüketim:** 48, 72, 168 ve 336 saatlik gecikmeler, son tam günün ortalaması ve bir önceki haftanın aynı saatinin ortalaması.

## Modeller

| Model | Rolü |
|---|---|
| Bir hafta önceki aynı saat | mevsimsel naif referans |
| Ridge regresyonu | aynı özelliklerle doğrusal referans |
| LightGBM (L1 amaç fonksiyonu, eğitim döneminin son iki ayında erken durdurma) | ana model |
| EPİAŞ ertesi gün yük planı | resmî karşılaştırma ölçütü |

Hatalar, ayrılmış yıl üzerinde MAPE ve MAE olarak raporlanır.

## Sonuçlar

Test dönemi: 24 Kasım 2025 – 23 Kasım 2026, dört tahminin de bulunduğu 8.735 saat. MAPE ortalama mutlak yüzde hata, MAE ise MWh cinsinden.

| Model | MAPE | MAE | Tatillerde MAPE |
|---|---|---|---|
| Bir hafta önceki aynı saat | %5,06 | 1.983 | %18,87 |
| Ridge regresyonu | %3,35 | 1.341 | %7,22 |
| **LightGBM** | **%2,27** | **933** | **%4,20** |
| EPİAŞ ertesi gün yük planı | %2,60 | 1.045 | %4,27 |

"Tatiller" resmî tatilleri, köprü günlerini ve bayram günlerini kapsar.

LightGBM, 364 günün %58'inde plandan daha düşük hata verdi. Günlük MAPE ortalaması plandan 0,33 puan düşük; bootstrap %95 aralığı 0,16 ile 0,50 arası (günler bütün olarak yeniden örneklendi ve birbirinden bağımsız sayıldı). 13 ayın 3'ünde plandan kötü kaldı: Mart, Ağustos ve Kasım 2026 (ilk ve son ay kısmi). En önemli özellikler bir hafta önceki aynı saat (modelin kazancının %42'si) ve iki hafta önceki aynı saat (%15).

![Sıradan bir hafta ve bir tatil haftası](reports/figures/forecast_weeks.png)

![Test yılında aylık hata](reports/figures/monthly_errors.png)

## Karşılaştırma ne kadar adil

Modellerin gördüğü sıcaklık, gerçekten yaşanan sıcaklık (Open-Meteo arşivi). Gerçek bir ertesi gün tahmininde ise yalnızca *tahmin edilmiş* sıcaklık bilinir; yani modellerin, yük planında olmayan bir avantajı var. Bunun ne kadar önemli olduğunu görmek için `src/sensitivity.py`, test dönemindeki sıcaklığa rastgele bir hata ekliyor (günde bir hata; boyutlar ölçülmüş tahmin hataları değil, varsayım) ve ayrıca modeli hiçbir hava durumu özelliği olmadan yeniden eğitiyor:

| LightGBM varyantı | MAPE |
|---|---|
| Gerçekleşen sıcaklık (ana sonuç) | %2,27 |
| Günlük 1 °C hata eklenmiş sıcaklık (standart sapma) | %2,32 |
| Günlük 2 °C hata eklenmiş sıcaklık | %2,45 |
| Günlük 3 °C hata eklenmiş sıcaklık | %2,58 |
| Hava durumu özelliği yok | %2,50 |
| EPİAŞ yük planı | %2,60 |

Yani modelin plana karşı üstünlüğü gerçek ama küçük; sıcaklık tahmini yaklaşık 3 °C şaşarsa üstünlük kayboluyor. Hava durumu olmayan varyant da planı hâlâ az farkla geçiyor; bu, sinyalin çoğunu takvim ve yakın geçmiş tüketim özelliklerinin taşıdığını düşündürüyor. Bu sayılar rastgele çekimler arasında yaklaşık 0,1 oynuyor.

## Sınırlar

- Tek bir çalıştırmadan tek bir test yılı. 0,33 puanlık kazanç, aylık hataların oynaklığının yanında küçük.
- Yukarıda anlatılan hava durumu avantajı. Adil bir test için arşivlenmiş ertesi gün sıcaklık tahminleri gerekir; bu projede yok.
- Model parametreleri önceden belirlendi, test yılında ayarlanmadı. Yalnızca ağaç sayısı, eğitim döneminin son iki ayında seçildi.
- Tüketim, EPİAŞ'ın gerçek zamanlı serisi; sonradan revize edilebilir.
- EPİAŞ planının hangi bilgilere dayandığını ve tam olarak ne zaman yayımlandığını bilmiyorum.

## Boru hattı

```
src/fetch_consumption.py   EPİAŞ'tan saatlik gerçek zamanlı tüketim ve yük planı (eptr2 ile)
src/fetch_weather.py       Open-Meteo'dan altı il için saatlik sıcaklık
src/features.py            saat başına bir satır: takvim, tatil, hava durumu ve gecikme özellikleri
src/train.py               modelleri eğitir ve karşılaştırır, metrikleri yazar
src/report.py             bu README için grafikler
src/sensitivity.py        hava durumu hatası ve hava durumsuz kontroller, plana karşı eşleştirilmiş karşılaştırma
app/streamlit_app.py       test yılının tahminlerini gün gün incelemek için
```

## Çalıştırma

[EPİAŞ Şeffaflık Platformu](https://kayit.epias.com.tr/epias-transparency-platform-registration-form)'nda ücretsiz bir hesap gerekir. `.env.example` dosyasını `.env` olarak kopyalayıp kendi bilgilerinizle doldurun. `.env` depoya gönderilmez.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.fetch_consumption
.venv/bin/python -m src.fetch_weather
.venv/bin/python -m src.features
.venv/bin/python -m src.train
.venv/bin/python -m src.report
.venv/bin/python -m src.sensitivity
.venv/bin/streamlit run app/streamlit_app.py
```

## Veri

- Tüketim ve yük planı: EPİAŞ Şeffaflık Platformu, Ocak 2019'dan itibaren.
- Hava durumu: Open-Meteo tarihsel arşivi.
- Tatiller: `holidays` Python paketi.

## İletişim

Begüm Çetin · begumcetin22@gmail.com · [GitHub](https://github.com/beguumcetin)
