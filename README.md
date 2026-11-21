English | [Türkçe](README.tr.md)

# Türkiye Day-Ahead Electricity Demand Forecast

An hourly, day-ahead forecast of Türkiye's national electricity consumption, compared against the official day-ahead load plan published by EPİAŞ (the Turkish energy exchange). The question: can a model built from calendar, holiday, weather and recent-consumption features beat the plan the grid operator already publishes?

**Status: work in progress.** The code is written, but the pipeline has not yet been run on the consumption data, which needs an EPİAŞ Transparency Platform account. This README therefore reports no results. They will be added once the models have been trained and tested.

## Design

- **Day-ahead setting.** The forecast for day D is made on the morning of D-1, when the latest complete day of consumption data is D-2. Every lag feature is therefore at least 48 hours old, so the model never sees information it would not have had in practice.
- **Honest evaluation.** The last 12 months are held out as the test period and everything before is used for training. Nothing from the test period is used to fit or tune a model.
- **A benchmark to beat.** EPİAŞ's own day-ahead load plan is scored on the same hours as the models.

## Features

- **Calendar:** hour, day of week, month, a sine/cosine pair for the day of the year, weekend flag.
- **Holidays** (from the `holidays` package for Türkiye): public holidays, religious holidays (bayram), half-day holidays that start at 13:00, bridge days (a working day squeezed between a holiday and a weekend) and the distance in days to the nearest bayram, capped at 7, because consumption stays low for several days around the long holidays.
- **Weather:** hourly temperature from the Open-Meteo archive for the six most populous provinces (Istanbul, Ankara, İzmir, Bursa, Antalya, Adana), combined into one national series weighted by population. From it come heating and cooling degrees (base 18 °C and 22 °C) and a 24-hour mean.
- **Recent consumption:** lags of 48, 72, 168 and 336 hours, the mean of the last full day, and the mean of the same hour over the previous week.

## Models

| Model | Role |
|---|---|
| Same hour one week earlier | seasonal-naive baseline |
| Ridge regression | linear baseline on the same features |
| LightGBM (L1 objective, early stopping on the last two months of the training period) | the main model |
| EPİAŞ day-ahead load plan | the official benchmark |

Errors are reported as MAPE and MAE on the held-out year.

## Pipeline

```
src/fetch_consumption.py   hourly real-time consumption and the load plan from EPİAŞ (via eptr2)
src/fetch_weather.py       hourly temperature for six provinces from Open-Meteo
src/features.py            one row per hour: calendar, holiday, weather and lag features
src/train.py               trains and compares the models, writes the metrics
src/report.py             the figures for this README
app/streamlit_app.py       explore the test-year forecasts day by day
```

## Run it

You need a free account on the [EPİAŞ Transparency Platform](https://kayit.epias.com.tr/epias-transparency-platform-registration-form). Copy `.env.example` to `.env` and fill in your own credentials. `.env` is not committed.

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

## Data

- Consumption and the load plan: EPİAŞ Transparency Platform, from January 2019.
- Weather: Open-Meteo historical archive.
- Holidays: the `holidays` Python package.
