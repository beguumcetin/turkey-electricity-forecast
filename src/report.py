"""Make the figures used in the README.

    python -m src.report
"""

import json

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, PROCESSED_DIR, ROOT

COLORS = {
    "actual": "#1f2937",
    "lightgbm": "#0f766e",
    "epias_plan": "#d97706",
    "ridge": "#6366f1",
    "seasonal_naive": "#9ca3af",
}
LABELS = {
    "actual": "Actual",
    "lightgbm": "LightGBM",
    "epias_plan": "EPİAŞ load plan",
    "ridge": "Ridge",
    "seasonal_naive": "Same hour last week",
}

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)


def daily_history(df):
    daily = df["consumption"].resample("D").sum() / 1000  # GWh per day
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(daily.index, daily, lw=0.6, color="#94a3b8")
    ax.plot(daily.index, daily.rolling(30, center=True).mean(), lw=1.8, color=COLORS["lightgbm"])
    ax.set_ylabel("GWh per day")
    ax.set_title("Daily electricity consumption in Türkiye (30-day average in green)")
    save(fig, "daily_history.png")


def hourly_profiles(df):
    groups = {
        "Weekday": df[(df.is_weekend == 0) & (df.is_holiday == 0)],
        "Sunday": df[(df.dayofweek == 6) & (df.is_holiday == 0)],
        "Bayram": df[df.is_bayram == 1],
    }
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for (name, g), color in zip(groups.items(), ["#0f766e", "#6366f1", "#d97706"]):
        ax.plot(g.groupby("hour")["consumption"].mean() / 1000, label=name, color=color, lw=2)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average GWh")
    ax.set_xticks(range(0, 24, 3))
    ax.set_title("Average daily load shape")
    ax.legend(frameon=False)
    save(fig, "hourly_profiles.png")


def temperature_curve(df):
    workdays = df[(df.is_weekend == 0) & (df.is_holiday == 0)]
    daily = workdays.resample("D").agg({"consumption": "sum", "temp_national": "mean"}).dropna()
    daily = daily[daily.consumption > 0]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sc = ax.scatter(daily.temp_national, daily.consumption / 1000, s=5, alpha=0.5, c=daily.index.year, cmap="viridis")
    ax.set_xlabel("Average temperature (°C, weighted over 6 cities)")
    ax.set_ylabel("GWh per working day")
    ax.set_title("Cold and hot days both push demand up")
    fig.colorbar(sc, ax=ax, label="Year")
    save(fig, "temperature_curve.png")


def forecast_weeks(pred):
    # One ordinary week and the week with the most holiday hours in the test period.
    weekly_hol = pred["is_holiday"].resample("W-SUN").sum()
    holiday_week_end = weekly_hol.idxmax()
    normal_week_end = weekly_hol[weekly_hol == 0].index[len(weekly_hol[weekly_hol == 0]) // 2]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharey=True)
    for ax, end, title in [(axes[0], normal_week_end, "An ordinary week"), (axes[1], holiday_week_end, "A holiday week")]:
        w = pred.loc[end - pd.Timedelta(days=6, hours=23): end + pd.Timedelta(hours=23, minutes=59)]
        for col in ["actual", "epias_plan", "lightgbm"]:
            ax.plot(w.index, w[col] / 1000, label=LABELS[col], color=COLORS[col], lw=2 if col == "actual" else 1.6,
                    ls="-" if col != "epias_plan" else "--")
        ax.set_title(f"{title}: {w.index.min():%d %b} – {w.index.max():%d %b %Y}")
        ax.set_ylabel("GWh")
    axes[0].legend(frameon=False, ncol=3)
    save(fig, "forecast_weeks.png")


def monthly_errors(metrics):
    m = pd.DataFrame(metrics["monthly_MAPE"]).T
    cols = ["seasonal_naive", "ridge", "epias_plan", "lightgbm"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = range(len(m))
    width = 0.2
    for i, c in enumerate(cols):
        ax.bar([p + (i - 1.5) * width for p in x], m[c], width, label=LABELS[c], color=COLORS[c])
    ax.set_xticks(list(x))
    ax.set_xticklabels(m.index, rotation=45, ha="right")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Monthly error on the test year (lower is better)")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
    ax.legend(frameon=False, ncol=4)
    save(fig, "monthly_errors.png")


def main():
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    pred = pd.read_parquet(PROCESSED_DIR / "predictions.parquet")
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    daily_history(df)
    hourly_profiles(df)
    temperature_curve(df)
    forecast_weeks(pred)
    monthly_errors(metrics)
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
