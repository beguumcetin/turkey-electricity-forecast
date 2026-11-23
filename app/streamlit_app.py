"""Explore the test-year forecasts day by day.

    streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "lightgbm": ("LightGBM (this project)", "#0f766e", "solid"),
    "epias_plan": ("EPİAŞ load plan", "#d97706", "dash"),
    "ridge": ("Ridge regression", "#6366f1", "dot"),
    "seasonal_naive": ("Same hour last week", "#9ca3af", "dot"),
}

st.set_page_config(page_title="Türkiye electricity forecast", page_icon="⚡", layout="wide")


@st.cache_data
def load():
    pred = pd.read_parquet(ROOT / "data" / "processed" / "predictions.parquet")
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    backtest_path = ROOT / "reports" / "backtest.json"
    backtest = json.loads(backtest_path.read_text()) if backtest_path.exists() else None
    return pred, metrics, backtest


pred, metrics, backtest = load()

st.title("⚡ Day-ahead electricity demand forecast for Türkiye")
st.caption(
    f"Hourly national consumption, test period {metrics['test_period'][0]} – {metrics['test_period'][1]}. "
    "Every forecast uses only what was known on the morning of the previous day: consumption at least 48 hours old "
    "and the temperature forecast made two days earlier."
)

cols = st.columns(len(MODELS))
for col, (key, (label, _, _)) in zip(cols, MODELS.items()):
    col.metric(label, f"{metrics['overall'][key]['MAPE']:.2f}% MAPE",
               help=f"Holidays only: {metrics['holidays_only'][key]['MAPE']:.2f}%")

left, right = st.columns([1, 3])
with left:
    days = sorted(set(pred.index.date))
    start = st.date_input("Start day", value=days[len(days) // 2], min_value=days[0], max_value=days[-1])
    n_days = st.slider("Number of days", 1, 14, 7)
    shown = st.multiselect("Models", list(MODELS), default=["lightgbm", "epias_plan"],
                           format_func=lambda k: MODELS[k][0])

window = pred.loc[str(start): str(pd.Timestamp(start) + pd.Timedelta(days=n_days - 1))]

with right:
    fig = go.Figure()
    fig.add_scatter(x=window.index, y=window["actual"] / 1000, name="Actual", line=dict(color="#111827", width=2.5))
    for key in shown:
        label, color, dash = MODELS[key]
        fig.add_scatter(x=window.index, y=window[key] / 1000, name=label, line=dict(color=color, dash=dash))
    fig.update_layout(yaxis_title="GWh", height=430, margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")

if len(window):
    err = pd.DataFrame(
        {MODELS[k][0]: [(abs(window[k] - window.actual) / window.actual).mean() * 100] for k in MODELS}
    ).T.rename(columns={0: "MAPE in this window (%)"}).round(2)
    st.dataframe(err, width="content")

with st.expander("What drives the LightGBM forecast?"):
    imp = pd.Series(metrics["feature_importance_gain_share"]).head(12)[::-1] * 100
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#0f766e"))
    fig.update_layout(xaxis_title="Share of total gain (%)", height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

if backtest:
    with st.expander("Three test years: does the result hold?", expanded=True):
        years = [f"{f['test_period'][0][:4]}–{f['test_period'][1][2:4]}" for f in backtest["folds"]]
        fig = go.Figure()
        for key in ["lightgbm", "epias_plan"]:
            label, color, _ = MODELS[key]
            fig.add_bar(x=years, y=[f["MAPE"][key] for f in backtest["folds"]], name=label, marker_color=color,
                        text=[f"{f['MAPE'][key]:.2f}%" for f in backtest["folds"]], textposition="outside")
        fig.update_layout(yaxis_title="MAPE (%)", height=330, margin=dict(t=10, b=10), barmode="group",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
        pooled = backtest["pooled"]
        lo, hi = pooled["weekly_block_bootstrap_95ci_pp"]
        st.caption(
            f"Each year is forecast by a model trained only on the years before it. Over all {pooled['days']} days "
            f"LightGBM's daily error is {pooled['mean_daily_mape_gain_pp']:.2f} points lower than the plan's "
            f"(95% interval {lo:.2f} to {hi:.2f}). The plan has improved every year and in the last one the gap "
            "is within the noise."
        )
