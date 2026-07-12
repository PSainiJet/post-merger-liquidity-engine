"""Treasury desk dashboard — 48-hour liquidity risk monitor."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import load_config
from src.prediction.liquidity_predictor import LiquidityEngine

ALERT_COLORS = {
    "OK": "#22c55e",
    "WARNING": "#f59e0b",
    "CRITICAL": "#ef4444",
}


def _render_alert_banner(at_risk: list[str], forecast: pd.DataFrame) -> None:
    critical = forecast[forecast["alert_level"] == "CRITICAL"]
    warning = forecast[forecast["alert_level"] == "WARNING"]

    if len(critical) > 0:
        st.error(
            f"CRITICAL liquidity shortfall risk: {', '.join(critical['currency'].tolist())} "
            f"— action required within 48 hours"
        )
    elif len(warning) > 0:
        st.warning(
            f"WARNING: {', '.join(warning['currency'].tolist())} approaching liquidity thresholds"
        )
    else:
        st.success("All monitored currencies (CHF, USD, EUR) within safe liquidity buffers.")


def _render_currency_cards(forecast: pd.DataFrame) -> None:
    cols = st.columns(len(forecast))
    for col, (_, row) in zip(cols, forecast.iterrows()):
        color = ALERT_COLORS[row["alert_level"]]
        with col:
            st.markdown(
                f"""
                <div style="border-left: 4px solid {color}; padding: 12px; background: #1e1e2e; border-radius: 8px;">
                    <h3 style="margin:0; color: {color};">{row['currency']}</h3>
                    <p style="font-size: 1.5rem; margin: 8px 0;">
                        {row['predicted_liquidity_48h_m']:,.0f} M
                    </p>
                    <p style="margin:0; color: #aaa;">48h forecast</p>
                    <p style="margin:4px 0 0; color: {color}; font-weight: bold;">
                        {row['alert_level']} — Buffer {row['buffer_pct']:.1f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_forecast_chart(forecast: pd.DataFrame) -> None:
    fig = go.Figure()
    for _, row in forecast.iterrows():
        fig.add_trace(
            go.Bar(
                name=f"{row['currency']} — RF",
                x=[f"{row['currency']} RF"],
                y=[row["rf_prediction_m"]],
                marker_color="#6366f1",
            )
        )
        fig.add_trace(
            go.Bar(
                name=f"{row['currency']} — TS",
                x=[f"{row['currency']} TS"],
                y=[row["ts_prediction_m"]],
                marker_color="#8b5cf6",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[row["currency"]],
                y=[row["min_required_m"]],
                mode="markers",
                marker=dict(size=14, color="#ef4444", symbol="line-ns-open"),
                name=f"{row['currency']} Min Required",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Ensemble Forecast vs Minimum Required Liquidity (M)",
        barmode="group",
        template="plotly_dark",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_stress_results(stress_results: dict) -> None:
    st.subheader("Stress Test Scenarios")
    for name, result in stress_results.items():
        with st.expander(f"Scenario: {name.replace('_', ' ').title()}", expanded=False):
            impact = result.liquidity_impact[
                ["currency", "liquidity_drop_m", "alert_level_baseline", "alert_level_stressed"]
            ]
            st.dataframe(impact, use_container_width=True)


def run_dashboard(project_root: Path | None = None) -> None:
    st.set_page_config(
        page_title="Global Intraday Liquidity Predictor",
        page_icon="🏦",
        layout="wide",
    )

    st.title("Global Intraday Liquidity Predictor & Stress Tester")
    st.caption(
        "Post-merger treasury consolidation monitor — 48-hour liquidity shortfall alerts"
    )

    cfg = load_config()
    root = project_root or Path(__file__).resolve().parent.parent.parent

    with st.sidebar:
        st.header("Controls")
        regenerate = st.checkbox("Regenerate simulated data", value=False)
        run_stress = st.checkbox("Run stress tests", value=True)
        refresh = st.button("Refresh Forecast")

        st.divider()
        st.markdown("**Monitored Currencies**")
        for ccy in cfg["project"]["currencies"]:
            st.markdown(f"- {ccy}")
        st.markdown(f"**Horizon:** {cfg['project']['horizon_hours']} hours")

    if refresh or "engine" not in st.session_state:
        with st.spinner("Loading data and running ensemble model..."):
            engine = LiquidityEngine(config=cfg, project_root=root)
            engine.initialize_data(regenerate=regenerate)

            if not (root / cfg["paths"]["model_artifacts"] / "random_forest.joblib").exists():
                engine.train()
            else:
                engine.load_models()

            st.session_state["engine"] = engine
            st.session_state["report"] = engine.get_report()
            if run_stress:
                st.session_state["stress"] = engine.run_stress_tests()

    report = st.session_state["report"]
    forecast = report.currency_forecasts

    st.markdown(f"*Last updated: {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}*")

    _render_alert_banner(report.at_risk_currencies, forecast)
    _render_currency_cards(forecast)

    col1, col2 = st.columns([2, 1])
    with col1:
        _render_forecast_chart(forecast)
    with col2:
        st.subheader("Recommended Actions")
        for action in report.recommended_actions:
            st.markdown(f"- {action}")

    st.subheader("Detailed Forecast")
    st.dataframe(
        forecast.style.map(
            lambda v: f"color: {ALERT_COLORS.get(v, 'white')}",
            subset=["alert_level"],
        ),
        use_container_width=True,
    )

    if "stress" in st.session_state:
        _render_stress_results(st.session_state["stress"])


if __name__ == "__main__":
    run_dashboard()
