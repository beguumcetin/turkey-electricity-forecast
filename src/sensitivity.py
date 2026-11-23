"""How fair is the comparison with the EPİAŞ load plan?

The models use *observed* temperature (Open-Meteo archive), while a real day-ahead
forecast only has a *forecast* temperature. This script measures how much that matters:
  1. the main LightGBM model (observed temperature)
  2. the same model with a random daily error added to the test-period temperature
     (one error per day, shared by its hours; the sizes are assumptions, not measured
     forecast errors)
  3. LightGBM retrained without any weather features
and also a paired daily comparison against the load plan.

    python -m src.sensitivity
"""

import json

import numpy as np
import pandas as pd

import src.train as T
from src.config import PROCESSED_DIR, ROOT

WEATHER = ["temp_national", "hdd", "cdd", "temp_24h_mean"]
ERROR_SDS = (1.0, 2.0, 3.0)


def main():
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    train, test = T.split(df)
    test = test[test["epias_plan"].notna()]
    out = {"hours_scored": int(len(test))}

    def score(pred):
        p = pd.Series(pred, index=test.index)
        return {"MAPE": round(T.mape(test[T.TARGET], p), 2), "MAE": round(T.mae(test[T.TARGET], p))}

    out["epias_plan"] = score(test["epias_plan"].to_numpy())
    model, n_trees = T.fit_lgbm(train)
    base_pred = model.predict(test[T.FEATURES])
    out["lightgbm_observed_temperature"] = score(base_pred)

    rng = np.random.default_rng(0)
    day = test.index.normalize()
    for sd in ERROR_SDS:
        t2 = test.copy()
        err = pd.Series(rng.normal(0, sd, day.nunique()), index=day.unique())
        t2["temp_national"] = t2["temp_national"] + err.reindex(day).to_numpy()
        t2["hdd"] = (18 - t2["temp_national"]).clip(lower=0)
        t2["cdd"] = (t2["temp_national"] - 22).clip(lower=0)
        t2["temp_24h_mean"] = t2["temp_national"].rolling(24, min_periods=12).mean()
        out[f"lightgbm_temperature_error_sd_{sd:g}C"] = score(model.predict(t2[T.FEATURES]))

    T.NUMERIC = [c for c in T.NUMERIC if c not in WEATHER]
    T.FEATURES = T.CATEGORICAL + T.NUMERIC
    no_weather, _ = T.fit_lgbm(train)
    out["lightgbm_no_weather"] = score(no_weather.predict(test[T.FEATURES]))

    # Paired daily comparison, main model vs the load plan.
    ape = lambda p: (p - test[T.TARGET]).abs() / test[T.TARGET] * 100
    daily = pd.DataFrame({"lgb": ape(pd.Series(base_pred, index=test.index)), "plan": ape(test["epias_plan"])})
    daily = daily.groupby(daily.index.normalize()).mean()
    gain = (daily["plan"] - daily["lgb"]).to_numpy()
    boot = np.random.default_rng(1).choice(gain, (5000, len(gain))).mean(axis=1)
    out["daily_comparison"] = {
        "days": int(len(daily)),
        "share_days_lightgbm_better": round(float((daily["lgb"] < daily["plan"]).mean()), 3),
        "mean_mape_gain_pp": round(float(gain.mean()), 3),
        "bootstrap_95ci_pp": [round(float(x), 3) for x in np.percentile(boot, [2.5, 97.5])],
    }

    (ROOT / "reports" / "sensitivity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
