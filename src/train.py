"""Train and compare day-ahead forecasts on the last 12 months.

Models:
  - seasonal naive: same hour one week earlier
  - ridge regression on the same features
  - LightGBM
  - EPİAŞ's own load plan (official day-ahead forecast), as the benchmark to beat

    python -m src.train
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODELS_DIR, PROCESSED_DIR, ROOT, TEST_MONTHS

TARGET = "consumption"
CATEGORICAL = ["hour", "dayofweek", "month"]
NUMERIC = [
    "doy_sin", "doy_cos", "is_weekend", "is_holiday", "is_bayram", "is_half_day",
    "is_bridge", "days_to_bayram", "temp_national", "hdd", "cdd", "temp_24h_mean",
    "lag_48h", "lag_72h", "lag_168h", "lag_336h", "mean_last_day", "mean_last_week_same_hour",
]
FEATURES = CATEGORICAL + NUMERIC


def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def split(df: pd.DataFrame):
    test_start = df.index.max().normalize() - pd.DateOffset(months=TEST_MONTHS) + pd.Timedelta(days=1)
    return df[df.index < test_start], df[df.index >= test_start]


def fit_ridge(train):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    model = make_pipeline(pre, Ridge(alpha=1.0))
    return model.fit(train[FEATURES], train[TARGET])


def fit_lgbm(train):
    # Hold out the last two months of the training period for early stopping.
    cut = train.index.max() - pd.DateOffset(months=2)
    fit, val = train[train.index <= cut], train[train.index > cut]
    params = dict(
        objective="regression_l1",  # robust to the odd outlier hour
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        n_estimators=5000,
        verbose=-1,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(
        fit[FEATURES], fit[TARGET],
        eval_X=(val[FEATURES],), eval_y=(val[TARGET],),
        callbacks=[lgb.early_stopping(200, verbose=False)],
        categorical_feature=CATEGORICAL,
    )
    # Refit on the whole training period with the number of trees we found.
    best = model.best_iteration_
    final = lgb.LGBMRegressor(**{**params, "n_estimators": best})
    final.fit(train[FEATURES], train[TARGET], categorical_feature=CATEGORICAL)
    return final, best


def main():
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    train, test = split(df)
    print(f"Train: {train.index.min():%Y-%m-%d} – {train.index.max():%Y-%m-%d} ({len(train)} h)")
    print(f"Test:  {test.index.min():%Y-%m-%d} – {test.index.max():%Y-%m-%d} ({len(test)} h)")

    ridge = fit_ridge(train)
    lgbm, n_trees = fit_lgbm(train)

    pred = pd.DataFrame({"actual": test[TARGET]}, index=test.index)
    pred["seasonal_naive"] = test["lag_168h"]
    pred["ridge"] = ridge.predict(test[FEATURES])
    pred["lightgbm"] = lgbm.predict(test[FEATURES])
    pred["epias_plan"] = test["epias_plan"]
    pred["is_holiday"] = test["is_holiday"] | test["is_bridge"] | (test["days_to_bayram"] == 0)

    models = ["seasonal_naive", "ridge", "lightgbm", "epias_plan"]
    scored = pred.dropna(subset=models)
    metrics = {
        "test_period": [f"{test.index.min():%Y-%m-%d}", f"{test.index.max():%Y-%m-%d}"],
        "lightgbm_trees": int(n_trees),
        "overall": {m: {"MAPE": mape(scored.actual, scored[m]), "MAE": mae(scored.actual, scored[m])} for m in models},
        "holidays_only": {
            m: {"MAPE": mape(h.actual, h[m])} for h in [scored[scored.is_holiday == 1]] for m in models
        },
    }
    monthly = (
        scored.assign(month=scored.index.strftime("%Y-%m"))
        .groupby("month")
        .apply(lambda g: pd.Series({m: mape(g.actual, g[m]) for m in models}), include_groups=False)
        .round(2)
    )
    metrics["monthly_MAPE"] = monthly.to_dict(orient="index")

    importance = (
        pd.Series(lgbm.booster_.feature_importance("gain"), index=FEATURES)
        .sort_values(ascending=False)
    )
    metrics["feature_importance_gain_share"] = (importance / importance.sum()).round(4).to_dict()

    pred.to_parquet(PROCESSED_DIR / "predictions.parquet")
    MODELS_DIR.mkdir(exist_ok=True)
    lgbm.booster_.save_model(str(MODELS_DIR / "lightgbm.txt"))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print("\nTest MAPE (%):")
    for m in models:
        print(f"  {m:15s} {metrics['overall'][m]['MAPE']:.2f}   (holidays: {metrics['holidays_only'][m]['MAPE']:.2f})")


if __name__ == "__main__":
    main()
