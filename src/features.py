"""Build the modelling table: one row per hour with calendar, holiday, weather
and lag features.

The setup is day-ahead: the forecast for day D is made on the morning of D-1,
when the latest complete day of consumption data is D-2. So every lag is at
least 48 hours old.

    python -m src.features
"""

import holidays
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR

LAGS = [48, 72, 168, 336]
BAYRAM_WORDS = ("Ramazan", "Kurban")


def holiday_table(years) -> pd.DataFrame:
    full = holidays.TR(years=years)
    half = holidays.TR(years=years, categories=("half_day",))
    days = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D")
    df = pd.DataFrame(index=days.date)
    df["is_holiday"] = [d in full for d in df.index]
    df["is_bayram"] = [any(w in full.get(d, "") for w in BAYRAM_WORDS) for d in df.index]
    df["is_half_day"] = [d in half for d in df.index]

    # Bridge day: a working day squeezed between a holiday and a weekend.
    # The government often makes these official, and either way many people take them off.
    wd = pd.Series(pd.to_datetime(df.index).day_of_week, index=df.index)
    off = df["is_holiday"] | (wd >= 5)
    prev_off, next_off = off.shift(1, fill_value=False), off.shift(-1, fill_value=False)
    df["is_bridge"] = ~off & prev_off & next_off

    # Distance to the nearest bayram day, capped at 7. Consumption drops for
    # several days around the long religious holidays, not just on them.
    bayram_days = pd.to_datetime([d for d in df.index if df.at[d, "is_bayram"]])
    all_days = pd.to_datetime(df.index)
    dist = np.min(np.abs((all_days.values[:, None] - bayram_days.values[None, :]).astype("timedelta64[D]").astype(int)), axis=1)
    df["days_to_bayram"] = np.minimum(dist, 7)
    return df


WEATHER_COLS = ["temp_national", "hdd", "cdd", "temp_24h_mean"]


def weather_features(temp: pd.Series) -> pd.DataFrame:
    """Temperature and what is derived from it. Kept in one place so the same
    features can be rebuilt from a forecast temperature at evaluation time."""
    return pd.DataFrame(
        {
            "temp_national": temp,
            # Heating and cooling degrees: demand rises when it is cold (heating)
            # and even more when it is hot (air conditioning).
            "hdd": (18 - temp).clip(lower=0),
            "cdd": (temp - 22).clip(lower=0),
            "temp_24h_mean": temp.rolling(24, min_periods=12).mean(),
        },
        index=temp.index,
    )


def with_temperature(df: pd.DataFrame, temp_col: str) -> pd.DataFrame:
    """Copy of the feature table with the weather features rebuilt from another
    temperature column, e.g. an archived forecast instead of the observed value."""
    out = df.copy()
    for col, values in weather_features(df[temp_col]).items():
        out[col] = values
    return out


def build(cons: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = cons.set_index("time").join(weather.set_index("time")[["temp_national"]], how="left")
    df["temp_national"] = df["temp_national"].interpolate(limit=3)

    t = df.index
    df["hour"] = t.hour
    df["dayofweek"] = t.day_of_week
    df["month"] = t.month
    df["doy_sin"] = np.sin(2 * np.pi * t.day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * t.day_of_year / 365.25)
    df["is_weekend"] = (t.day_of_week >= 5).astype(int)

    hol = holiday_table(sorted(set(t.year)))
    day = pd.Index(t.date)
    for col in hol.columns:
        df[col] = hol[col].reindex(day).to_numpy().astype(int)
    # Half-day holidays only start at 13:00.
    df["is_half_day"] = df["is_half_day"] * (df["hour"] >= 13)

    for col, values in weather_features(df["temp_national"]).items():
        df[col] = values

    # Lags of the target. Shifting by hours assumes a gap-free hourly index,
    # which make_hourly() guarantees.
    y = df["consumption"]
    for lag in LAGS:
        df[f"lag_{lag}h"] = y.shift(lag)
    df["mean_last_day"] = y.shift(48).rolling(24).mean()
    df["mean_last_week_same_hour"] = sum(y.shift(24 * k) for k in range(2, 9)) / 7
    return df


def make_hourly(cons: pd.DataFrame) -> pd.DataFrame:
    """Put consumption on a complete hourly index and fill short gaps."""
    s = cons.set_index("time")["consumption"]
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(full).interpolate(limit=3)
    return s.rename_axis("time").reset_index()


def main():
    cons = make_hourly(pd.read_parquet(RAW_DIR / "consumption.parquet"))
    weather = pd.read_parquet(RAW_DIR / "weather.parquet")
    df = build(cons, weather)

    plan = pd.read_parquet(RAW_DIR / "load_plan.parquet").set_index("time")["epias_plan"]
    df["epias_plan"] = plan.reindex(df.index)

    # Archived temperature forecasts, used only at evaluation time (see train.py).
    fc_path = RAW_DIR / "weather_forecast.parquet"
    if fc_path.exists():
        fc = pd.read_parquet(fc_path).set_index("time")
        for col in fc.columns:
            df[col] = fc[col].reindex(df.index)

    df = df.dropna(subset=["consumption", f"lag_{max(LAGS)}h", "temp_national"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / "features.parquet")
    print(f"Feature table: {df.shape[0]} hours, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
