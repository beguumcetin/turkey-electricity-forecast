"""Download hourly temperature for the largest cities from the Open-Meteo archive
(free, no API key) and combine them into one weighted national series.

    python -m src.fetch_weather
"""

from datetime import date, timedelta

import pandas as pd
import requests

from src.config import CITIES, RAW_DIR, START_DATE, TIMEZONE

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_city(lat: float, lon: float, start: str, end: str) -> pd.Series:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m",
        "timezone": TIMEZONE,
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=120)
    r.raise_for_status()
    h = r.json()["hourly"]
    idx = pd.DatetimeIndex(pd.to_datetime(h["time"])).tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    return pd.Series(h["temperature_2m"], index=idx).loc[lambda s: s.index.notna()]


def main():
    end = (date.today() - timedelta(days=1)).isoformat()
    cols = {}
    for name, (lat, lon, _) in CITIES.items():
        cols[name] = fetch_city(lat, lon, START_DATE, end)
        print(f"{name}: {len(cols[name])} hours")
    df = pd.DataFrame(cols)
    df = df[~df.index.duplicated()]

    weights = pd.Series({name: w for name, (_, _, w) in CITIES.items()})
    df["temp_national"] = (df[weights.index] * weights).sum(axis=1) / weights.sum()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.rename_axis("time").reset_index().to_parquet(RAW_DIR / "weather.parquet", index=False)
    print(f"Saved weather to {RAW_DIR / 'weather.parquet'}")


if __name__ == "__main__":
    main()
