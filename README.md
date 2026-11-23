English | [Türkçe](README.tr.md)

# Türkiye Day-Ahead Electricity Demand Forecast

An hourly, day-ahead forecast of Türkiye's national electricity consumption, compared against the official day-ahead load plan published by EPİAŞ (the Turkish energy exchange). The question: can a model built from calendar, holiday, weather and recent-consumption features beat the plan the grid operator already publishes?

On a held-out year, the LightGBM model beats the load plan by a small margin (2.27% against 2.60% mean absolute percentage error), but the comparison favours the model somewhat; see [How fair is the comparison](#how-fair-is-the-comparison).

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

## Results

Test period: 24 November 2025 to 23 November 2026, 8,735 hours for which all four forecasts exist. MAPE is the mean absolute percentage error and MAE is in MWh.

| Model | MAPE | MAE | MAPE on holidays |
|---|---|---|---|
| Same hour one week earlier | 5.06% | 1,983 | 18.87% |
| Ridge regression | 3.35% | 1,341 | 7.22% |
| **LightGBM** | **2.27%** | **933** | **4.20%** |
| EPİAŞ day-ahead load plan | 2.60% | 1,045 | 4.27% |

"Holidays" means public holidays, bridge days and bayram days.

LightGBM had a lower error than the plan on 58% of the 364 days. Its average daily MAPE was 0.33 percentage points lower, with a bootstrap 95% interval of 0.16 to 0.50 (resampling whole days and treating them as independent). It did worse than the plan in 3 of the 13 months: March, August and November 2026 (the first and last months are partial). The most important features are the same hour one week earlier (42% of the model's gain) and two weeks earlier (15%).

![An ordinary week and a holiday week](reports/figures/forecast_weeks.png)

![Monthly error on the test year](reports/figures/monthly_errors.png)

## How fair is the comparison

The temperature the models see is the temperature that actually occurred (Open-Meteo archive). A real day-ahead forecast only has a *forecast* temperature, so the models have an advantage the load plan does not. To see how much that matters, `src/sensitivity.py` adds a random error to the test-period temperature (one error per day; the sizes are assumptions, not measured forecast errors) and also retrains the model without any weather feature:

| LightGBM variant | MAPE |
|---|---|
| Observed temperature (main result) | 2.27% |
| Temperature with a daily error of 1 °C (standard deviation) | 2.32% |
| Temperature with a daily error of 2 °C | 2.45% |
| Temperature with a daily error of 3 °C | 2.58% |
| No weather features | 2.50% |
| EPİAŞ load plan | 2.60% |

So the model's edge over the plan is real but small, and it shrinks to nothing if the temperature forecast is off by about 3 °C. The no-weather variant still edges the plan, which suggests calendar and recent-consumption features carry most of the signal. These numbers move by about 0.1 between random draws.

## Limits

- One test year from one run. The 0.33-point gain is small next to how much the monthly errors vary.
- The weather advantage described above. A fair test needs archived day-ahead temperature forecasts, which this project does not have.
- The model parameters were set in advance and not tuned on the test year. Only the number of trees was chosen, on the last two months of the training period.
- Consumption is the real-time series from EPİAŞ, which may be revised later.
- I do not know what information EPİAŞ's plan is built on or exactly when it is published.

## Pipeline

```
src/fetch_consumption.py   hourly real-time consumption and the load plan from EPİAŞ (via eptr2)
src/fetch_weather.py       hourly temperature for six provinces from Open-Meteo
src/features.py            one row per hour: calendar, holiday, weather and lag features
src/train.py               trains and compares the models, writes the metrics
src/report.py             the figures for this README
src/sensitivity.py        weather-error and no-weather checks, paired comparison with the plan
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
.venv/bin/python -m src.sensitivity
.venv/bin/streamlit run app/streamlit_app.py
```

## Data

- Consumption and the load plan: EPİAŞ Transparency Platform, from January 2019.
- Weather: Open-Meteo historical archive.
- Holidays: the `holidays` Python package.

## Contact

Begüm Çetin · begumcetin22@gmail.com · [GitHub](https://github.com/beguumcetin)
