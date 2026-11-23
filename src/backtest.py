"""Rolling-origin backtest: the same comparison on three separate test years.

For each test year the models are trained only on the data before it (expanding
window) and forecast with the temperature predicted two days earlier, exactly as in
train.py. Archived forecasts start in 2023, so the first test year begins in November
2023. Hours without an archived forecast (about 500 in December 2023 and January 2024)
are left out of that year's score.

    python -m src.backtest
"""

import json

import numpy as np
import pandas as pd

import src.train as T
from src.config import PROCESSED_DIR, ROOT
from src.features import with_temperature

TEST_YEAR_ENDS = ["2024-11-23", "2025-11-23", "2026-11-23"]


def run_fold(df: pd.DataFrame, end: str) -> tuple[dict, pd.DataFrame]:
    end_ts = pd.Timestamp(end, tz=df.index.tz) + pd.Timedelta(hours=23)
    start_ts = (end_ts.normalize() - pd.DateOffset(months=12) + pd.Timedelta(days=1))
    train = df[df.index < start_ts]
    test_idx = df.index[(df.index >= start_ts) & (df.index <= end_ts)]

    ridge = T.fit_ridge(train)
    lgbm, n_trees = T.fit_lgbm(train)
    fair = with_temperature(df, T.FAIR_TEMP).loc[test_idx]

    pred = pd.DataFrame({"actual": df.loc[test_idx, T.TARGET]})
    pred["seasonal_naive"] = df.loc[test_idx, "lag_168h"]
    pred["ridge"] = T.predict_rows(ridge, fair)
    pred["lightgbm"] = T.predict_rows(lgbm, fair)
    pred["epias_plan"] = df.loc[test_idx, "epias_plan"]
    pred = pred.dropna()

    ape = lambda c: (pred[c] - pred.actual).abs() / pred.actual * 100
    daily = pd.DataFrame({"lgb": ape("lightgbm"), "plan": ape("epias_plan")})
    daily = daily.groupby(daily.index.normalize()).mean()
    fold = {
        "test_period": [f"{start_ts:%Y-%m-%d}", end],
        "train_hours": int(len(train)),
        "hours_scored": int(len(pred)),
        "lightgbm_trees": int(n_trees),
        "MAPE": {m: round(T.mape(pred.actual, pred[m]), 2) for m in T.MODELS},
        "share_days_lightgbm_better": round(float((daily.lgb < daily.plan).mean()), 3),
    }
    return fold, daily


def main():
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    folds, dailies = [], []
    for end in TEST_YEAR_ENDS:
        fold, daily = run_fold(df, end)
        folds.append(fold)
        dailies.append(daily)
        print(f"{fold['test_period'][0]} – {fold['test_period'][1]}: {fold['MAPE']}  "
              f"(LightGBM better on {fold['share_days_lightgbm_better']:.0%} of days)")

    daily = pd.concat(dailies)
    gain = daily["plan"] - daily["lgb"]
    # Errors on neighbouring days are correlated, so resample whole weeks, not days.
    weekly = gain.groupby(gain.index.to_series().dt.strftime("%G-%V")).mean().to_numpy()
    boot = np.random.default_rng(1).choice(weekly, (10000, len(weekly))).mean(axis=1)
    pooled = {
        "days": int(len(daily)),
        "share_days_lightgbm_better": round(float((daily.lgb < daily.plan).mean()), 3),
        "mean_daily_mape_gain_pp": round(float(gain.mean()), 3),
        "weekly_block_bootstrap_95ci_pp": [round(float(x), 3) for x in np.percentile(boot, [2.5, 97.5])],
    }
    print("pooled:", pooled)
    (ROOT / "reports" / "backtest.json").write_text(json.dumps({"folds": folds, "pooled": pooled}, indent=2))


if __name__ == "__main__":
    main()
