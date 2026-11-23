"""Where does the main model go wrong on the test year, and why?

Writes reports/error_analysis.json with
- the worst days for LightGBM next to the load plan,
- how much of the error sits on holidays and the days around a bayram,
- how quickly consumption recovered after every bayram since 2019,
- the average signed error per month (does the model run high or low?).

    python -m src.error_analysis
"""

import json

import holidays
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, ROOT


def bayram_recovery(daily: pd.Series, years) -> list[dict]:
    """Consumption on the first two working days after each bayram, relative to the
    same weekdays two weeks earlier."""
    hol = holidays.TR(years=years)
    days = sorted(d for d, n in hol.items() if "Ramazan" in n or "Kurban" in n)
    groups, cur = [], []
    for d in days:
        if cur and (d - cur[-1]).days > 1:
            groups.append(cur)
            cur = []
        cur.append(d)
    groups.append(cur)

    rows = []
    for g in groups:
        d = pd.Timestamp(g[-1], tz="Europe/Istanbul")
        after = []
        while len(after) < 2:
            d += pd.Timedelta(days=1)
            if d.dayofweek < 5 and d.date() not in hol:
                after.append(d)
        ratios = [daily[a] / daily[a - pd.Timedelta(days=14)] for a in after
                  if a in daily.index and a - pd.Timedelta(days=14) in daily.index]
        if ratios:
            rows.append({"bayram": hol[g[0]].split()[0], "start": g[0].isoformat(),
                         "first_workday_after": after[0].date().isoformat(),
                         "ratio_to_two_weeks_before": round(float(np.mean(ratios)), 3)})
    return rows


def main():
    f = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    p = pd.read_parquet(PROCESSED_DIR / "predictions.parquet").dropna(subset=["lightgbm", "epias_plan"])
    hol = holidays.TR(years=sorted(set(p.index.year)))

    ape = lambda c: (p[c] - p.actual).abs() / p.actual * 100
    day = p.index.normalize()
    d = pd.DataFrame({
        "lightgbm_mape": ape("lightgbm"), "plan_mape": ape("epias_plan"),
        "lightgbm_bias_pct": (p.lightgbm - p.actual) / p.actual * 100,
        "plan_bias_pct": (p.epias_plan - p.actual) / p.actual * 100,
    }).groupby(day).mean()
    extra = f.loc[p.index, ["is_bridge", "days_to_bayram", "temp_national"]].groupby(day).mean()
    d = d.join(extra)
    d["holiday"] = [hol.get(x.date(), "") for x in d.index]
    d["weekday"] = d.index.day_name()

    near = (d.holiday != "") | (d.is_bridge > 0) | (d.days_to_bayram <= 2)
    worst = d.sort_values("lightgbm_mape", ascending=False).head(10)

    daily_cons = f["consumption"].groupby(f.index.normalize()).mean()
    out = {
        "worst_days": [
            {"date": f"{i:%Y-%m-%d}", "weekday": r.weekday, "holiday": r.holiday,
             "lightgbm_mape": round(r.lightgbm_mape, 2), "plan_mape": round(r.plan_mape, 2),
             "lightgbm_bias_pct": round(r.lightgbm_bias_pct, 2)}
            for i, r in worst.iterrows()
        ],
        "holiday_window": {
            "days": int(near.sum()), "of_days": int(len(d)),
            "lightgbm_mape_on_them": round(float(d.lightgbm_mape[near].mean()), 2),
            "lightgbm_mape_other_days": round(float(d.lightgbm_mape[~near].mean()), 2),
            "plan_mape_other_days": round(float(d.plan_mape[~near].mean()), 2),
        },
        "bayram_recovery": bayram_recovery(daily_cons, range(2019, p.index.year.max() + 1)),
        "monthly_bias_pct": (
            d.groupby(d.index.strftime("%Y-%m"))[["lightgbm_bias_pct", "plan_bias_pct"]].mean().round(2)
            .to_dict(orient="index")
        ),
    }
    (ROOT / "reports" / "error_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ["holiday_window", "monthly_bias_pct"]}, indent=1, ensure_ascii=False))
    print("recovery ratios:", [(r["start"], r["ratio_to_two_weeks_before"]) for r in out["bayram_recovery"]])


if __name__ == "__main__":
    main()
