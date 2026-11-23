"""Two checks on the main result in reports/metrics.json.

1. How much do the weather features add? LightGBM is retrained without them.
2. Is the gain over the EPİAŞ load plan more than noise? Daily errors of the main
   LightGBM model (forecast temperature) and of the plan are compared day by day,
   with a bootstrap over days.

    python -m src.sensitivity
"""

import json

import numpy as np
import pandas as pd

import src.train as T
from src.config import PROCESSED_DIR, ROOT
from src.features import WEATHER_COLS


def main():
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    train, test = T.split(df)
    pred = pd.read_parquet(PROCESSED_DIR / "predictions.parquet").dropna(subset=T.MODELS + T.REFERENCES)
    out = {"hours_scored": int(len(pred))}

    # 1. No weather at all.
    T.NUMERIC = [c for c in T.NUMERIC if c not in WEATHER_COLS]
    T.FEATURES = T.CATEGORICAL + T.NUMERIC
    model, n_trees = T.fit_lgbm(train)
    no_weather = pd.Series(model.predict(test[T.FEATURES]), index=test.index).reindex(pred.index)
    out["MAPE"] = {
        "lightgbm_forecast_temp": round(T.mape(pred.actual, pred.lightgbm), 2),
        "lightgbm_observed_temp": round(T.mape(pred.actual, pred.lightgbm_observed_temp), 2),
        "lightgbm_no_weather": round(T.mape(pred.actual, no_weather), 2),
        "epias_plan": round(T.mape(pred.actual, pred.epias_plan), 2),
    }

    # 2. Paired daily comparison with the plan.
    ape = lambda p: (p - pred.actual).abs() / pred.actual * 100
    daily = pd.DataFrame({"lgb": ape(pred.lightgbm), "plan": ape(pred.epias_plan)})
    daily = daily.groupby(daily.index.normalize()).mean()
    gain = daily["plan"] - daily["lgb"]
    # Errors on neighbouring days are correlated, so resample whole weeks, not days.
    weekly = gain.groupby(gain.index.to_series().dt.strftime("%G-%V")).mean().to_numpy()
    boot = np.random.default_rng(1).choice(weekly, (10000, len(weekly))).mean(axis=1)
    out["daily_comparison"] = {
        "days": int(len(daily)),
        "share_days_lightgbm_better": round(float((daily["lgb"] < daily["plan"]).mean()), 3),
        "mean_daily_mape_gain_pp": round(float(gain.mean()), 3),
        "weekly_block_bootstrap_95ci_pp": [round(float(x), 3) for x in np.percentile(boot, [2.5, 97.5])],
    }

    (ROOT / "reports" / "sensitivity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
