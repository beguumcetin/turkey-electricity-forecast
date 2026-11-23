"""The day-ahead rule: the forecast for an hour may only use consumption that is at
least 48 hours old. These tests build features on synthetic data and check it."""

import numpy as np
import pandas as pd
import pytest

from src.features import WEATHER_COLS, build, weather_features, with_temperature
from src.train import FEATURES, split


def synthetic(days: int = 45, seed: int = 0):
    idx = pd.date_range("2024-03-01", periods=days * 24, freq="h", tz="Europe/Istanbul")
    rng = np.random.default_rng(seed)
    hour = np.arange(len(idx))
    cons = pd.DataFrame({
        "time": idx,
        "consumption": 30000 + 5000 * np.sin(hour * 2 * np.pi / 24) + rng.normal(0, 500, len(idx)),
    })
    weather = pd.DataFrame({"time": idx, "temp_national": 15 + 8 * np.sin(hour * 2 * np.pi / 24)})
    return cons, weather


@pytest.mark.parametrize("pos", [20 * 24, 30 * 24 + 7, 44 * 24 + 23])
def test_features_do_not_see_the_last_47_hours_or_the_future(pos):
    cons, weather = synthetic()
    t = cons["time"].iloc[pos]
    base = build(cons, weather)

    # Scramble every consumption value from t-47h onwards, including the future.
    changed = cons.copy()
    recent = changed["time"] > t - pd.Timedelta(hours=48)
    changed.loc[recent, "consumption"] = np.random.default_rng(1).uniform(0, 1e6, recent.sum())
    after = build(changed, weather)

    pd.testing.assert_series_equal(base.loc[t, FEATURES], after.loc[t, FEATURES], check_names=False)


def test_the_check_is_not_vacuous_48_hours_back_does_matter():
    cons, weather = synthetic()
    t = cons["time"].iloc[30 * 24]
    base = build(cons, weather)
    changed = cons.copy()
    changed.loc[changed["time"] == t - pd.Timedelta(hours=48), "consumption"] += 10000
    after = build(changed, weather)
    assert after.loc[t, "lag_48h"] == pytest.approx(base.loc[t, "lag_48h"] + 10000)


def test_split_keeps_the_test_year_after_training():
    idx = pd.date_range("2022-01-01", "2024-12-31 23:00", freq="h", tz="Europe/Istanbul")
    df = pd.DataFrame({"x": 1.0}, index=idx)
    train, test = split(df)
    assert train.index.max() < test.index.min()
    assert len(train) + len(test) == len(df)
    assert test.index.max() - test.index.min() < pd.Timedelta(days=366)


def test_degree_hours():
    w = weather_features(pd.Series([10.0, 20.0, 30.0]))
    assert w["hdd"].tolist() == [8.0, 0.0, 0.0]
    assert w["cdd"].tolist() == [0.0, 0.0, 8.0]


def test_swapping_temperature_only_touches_weather_features():
    cons, weather = synthetic()
    df = build(cons, weather)
    df["temp_fc"] = df["temp_national"] + 1.5
    swapped = with_temperature(df, "temp_fc")
    changed = [c for c in df.columns if not df[c].equals(swapped[c])]
    assert set(changed) <= set(WEATHER_COLS)
    assert np.allclose(swapped["temp_national"], df["temp_national"] + 1.5)
