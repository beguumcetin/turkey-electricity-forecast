[English](README.md) | Türkçe

# Türkiye Ertesi Gün Elektrik Talep Tahmini

Türkiye'nin ulusal elektrik tüketimi için saatlik, ertesi güne dönük bir tahmin; sonuçlar EPİAŞ'ın (Enerji Piyasaları İşletme A.Ş.) yayımladığı resmî ertesi gün yük planıyla karşılaştırılıyor. Soru şu: takvim, tatil, hava durumu ve yakın geçmiş tüketim özelliklerinden kurulan bir model, şebeke işletmecisinin zaten yayımladığı planı geçebilir mi?

**Durum: devam ediyor.** Kod yazıldı, ancak boru hattı henüz tüketim verisi üzerinde çalıştırılmadı; bu veri için bir EPİAŞ Şeffaflık Platformu hesabı gerekiyor. Bu yüzden bu README sonuç vermiyor. Modeller eğitilip test edildikten sonra eklenecek.

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

## Boru hattı

```
src/fetch_consumption.py   EPİAŞ'tan saatlik gerçek zamanlı tüketim ve yük planı (eptr2 ile)
src/fetch_weather.py       Open-Meteo'dan altı il için saatlik sıcaklık
src/features.py            saat başına bir satır: takvim, tatil, hava durumu ve gecikme özellikleri
src/train.py               modelleri eğitir ve karşılaştırır, metrikleri yazar
src/report.py             bu README için grafikler
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
.venv/bin/streamlit run app/streamlit_app.py
```

## Veri

- Tüketim ve yük planı: EPİAŞ Şeffaflık Platformu, Ocak 2019'dan itibaren.
- Hava durumu: Open-Meteo tarihsel arşivi.
- Tatiller: `holidays` Python paketi.
