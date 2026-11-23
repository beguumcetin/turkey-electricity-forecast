"""Download archived temperature *forecasts* for the same six provinces.

The main weather file holds the temperature that actually occurred, which a real
day-ahead forecaster would not have. Open-Meteo's Previous Runs API keeps what its
models predicted for each hour one and two days before that hour:

- `temperature_2m_previous_day2`: from the model run 48 hours earlier. For every
  hour of day D this run was finished before the morning of D-1, when the
  day-ahead forecast is made, so it is the honest choice and the one used for
  the main results.
- `temperature_2m_previous_day1`: from the run 24 hours earlier. For the later
  hours of D that run comes after the morning of D-1, so it is slightly
  optimistic. Kept as a reference.

    python -m src.fetch_weather_forecast
"""

import time
from datetime import date, timedelta

import pandas as pd
import requests

from src.config import CITIES, FORECAST_START, RAW_DIR, TIMEZONE

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARS = {"temperature_2m_previous_day1": "fc_d1", "temperature_2m_previous_day2": "fc_d2"}


def fetch_city(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(VARS),
        "start_date": start,
        "end_date": end,
        "timezone": TIMEZONE,
    }
    for attempt in range(4):
        r = requests.get(URL, params=params, timeout=180)
        if r.status_code != 429:
            break
        time.sleep(30 * (attempt + 1))
    r.raise_for_status()
    h = r.json()["hourly"]
    idx = pd.DatetimeIndex(pd.to_datetime(h["time"])).tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    df = pd.DataFrame({short: h[long] for long, short in VARS.items()}, index=idx)
    return df[df.index.notna()].astype(float)


def main():
    end = (date.today() - timedelta(days=1)).isoformat()
    weights = pd.Series({name: w for name, (_, _, w) in CITIES.items()})
    per_city = {}
    for name, (lat, lon, _) in CITIES.items():
        per_city[name] = fetch_city(lat, lon, FORECAST_START, end)
        cov = per_city[name].notna().mean().round(3).to_dict()
        print(f"{name}: {len(per_city[name])} hours, coverage {cov}")

    out = {}
    for short in VARS.values():
        city = pd.DataFrame({name: df[short] for name, df in per_city.items()})
        city = city[~city.index.duplicated()]
        # A national value only when every province has a forecast for that hour.
        out[f"temp_{short}"] = (city[weights.index] * weights).sum(axis=1, min_count=len(weights)) / weights.sum()
    df = pd.DataFrame(out).rename_axis("time").reset_index()
    df.to_parquet(RAW_DIR / "weather_forecast.parquet", index=False)
    print(f"Saved {len(df)} hours to {RAW_DIR / 'weather_forecast.parquet'}")


if __name__ == "__main__":
    main()
