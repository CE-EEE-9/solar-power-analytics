import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import is_datetime64_any_dtype

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_PATH, ANOMALY_THRESHOLD


DEFAULT_COLOR_SEQ = px.colors.qualitative.Set2


def load_clean_data(plant: int = 0) -> pd.DataFrame:
    """
    Load the cleaned dataset produced by src/data_loader.py.
    """
    file_path = os.path.join(PROCESSED_DATA_PATH, f"clean_plant{plant}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Cleaned file not found: {file_path}. Run src/data_loader.py first."
        )
    df = pd.read_csv(file_path)
    return _ensure_datetime(df)


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "DATE_TIME" not in df.columns:
        raise ValueError("DATE_TIME column is required for visualization.")
    df = df.copy()
    if not is_datetime64_any_dtype(df["DATE_TIME"]):
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], errors="coerce")
    if "DATE" not in df.columns:
        df["DATE"] = df["DATE_TIME"].dt.date
    if "HOUR" not in df.columns:
        df["HOUR"] = df["DATE_TIME"].dt.hour
    return df


def _ensure_source_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "SOURCE_KEY" not in df.columns:
        if "SOURCE_KEY_gen" in df.columns:
            df["SOURCE_KEY"] = df["SOURCE_KEY_gen"]
    return df


def _daily_energy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily energy per plant using DAILY_YIELD when available.
    """
    df = _ensure_datetime(df)
    df = _ensure_source_key(df)
    if "DAILY_YIELD" in df.columns and "SOURCE_KEY" in df.columns:
        daily = (
            df.groupby(["DATE", "SOURCE_KEY", "PLANT_ID"], as_index=False)["DAILY_YIELD"]
            .max()
            .groupby(["DATE", "PLANT_ID"], as_index=False)["DAILY_YIELD"]
            .sum()
            .rename(columns={"DAILY_YIELD": "DAILY_ENERGY"})
        )
    else:
        daily = (
            df.groupby(["DATE", "PLANT_ID"], as_index=False)["AC_POWER"]
            .sum()
            .rename(columns={"AC_POWER": "DAILY_ENERGY"})
        )
    return daily


def plot_daily_energy(df: pd.DataFrame) -> go.Figure:
    """
    Line chart of daily total energy by plant.
    """
    daily = _daily_energy(df)
    fig = px.line(
        daily,
        x="DATE",
        y="DAILY_ENERGY",
        color="PLANT_ID" if "PLANT_ID" in daily.columns else None,
        markers=True,
        color_discrete_sequence=DEFAULT_COLOR_SEQ,
        title="Daily Energy",
    )
    fig.update_layout(legend_title_text="Plant")
    return fig


def plot_inverter_comparison(
    df: pd.DataFrame, date: Optional[pd.Timestamp] = None, top_n: int = 10
) -> go.Figure:
    """
    Bar chart comparing inverter energy on a specific date.
    """
    df = _ensure_datetime(df)
    df = _ensure_source_key(df)
    if "SOURCE_KEY" not in df.columns:
        raise ValueError("SOURCE_KEY column is required for inverter comparison.")

    if date is None:
        date = pd.to_datetime(df["DATE"].max())
    date = pd.to_datetime(date).date()

    day_df = df[df["DATE"] == date]
    if "DAILY_YIELD" in day_df.columns:
        metrics = (
            day_df.groupby("SOURCE_KEY", as_index=False)["DAILY_YIELD"].max()
            .rename(columns={"DAILY_YIELD": "ENERGY"})
        )
    else:
        metrics = (
            day_df.groupby("SOURCE_KEY", as_index=False)["AC_POWER"].sum()
            .rename(columns={"AC_POWER": "ENERGY"})
        )

    metrics = metrics.sort_values("ENERGY", ascending=False).head(top_n)
    fig = px.bar(
        metrics,
        x="SOURCE_KEY",
        y="ENERGY",
        color="ENERGY",
        color_continuous_scale="viridis",
        title=f"Top {top_n} Inverters on {date}",
    )
    fig.update_layout(xaxis_title="Inverter", yaxis_title="Energy")
    return fig


def plot_irradiation_vs_power(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of irradiation vs AC power.
    """
    required = {"IRRADIATION", "AC_POWER"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Missing columns for scatter plot: {missing}")
    fig = px.scatter(
        df,
        x="IRRADIATION",
        y="AC_POWER",
        color="PLANT_ID" if "PLANT_ID" in df.columns else None,
        opacity=0.6,
        color_discrete_sequence=DEFAULT_COLOR_SEQ,
        title="Irradiation vs AC Power",
    )
    return fig


def plot_temperature_profile(df: pd.DataFrame) -> go.Figure:
    """
    Line chart for ambient and module temperatures.
    """
    df = _ensure_datetime(df)
    required = {"AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Missing temperature columns: {missing}")

    temp_df = (
        df.groupby("DATE_TIME", as_index=False)[
            ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]
        ]
        .mean()
        .melt("DATE_TIME", var_name="Metric", value_name="Temperature")
    )
    fig = px.line(
        temp_df,
        x="DATE_TIME",
        y="Temperature",
        color="Metric",
        color_discrete_sequence=DEFAULT_COLOR_SEQ,
        title="Temperature Profile",
    )
    return fig


def plot_anomaly_points(
    df: pd.DataFrame,
    threshold: float = ANOMALY_THRESHOLD,
    window: int = 96,
) -> go.Figure:
    """
    Highlight low-power points relative to rolling median per inverter.
    """
    df = _ensure_datetime(df)
    df = _ensure_source_key(df)
    if "SOURCE_KEY" not in df.columns:
        raise ValueError("SOURCE_KEY column is required for anomaly detection.")

    df = df.sort_values("DATE_TIME").reset_index(drop=True)
    rolling = (
        df.groupby("SOURCE_KEY")["AC_POWER"]
        .rolling(window=window, min_periods=max(4, window // 4))
        .median()
        .reset_index(level=0, drop=True)
    )

    expected = rolling.replace(0, np.nan)
    ratio = df["AC_POWER"] / expected
    mask = (ratio < threshold).fillna(False)
    anomalies = df.loc[mask].copy()

    fig = px.scatter(
        df,
        x="DATE_TIME",
        y="AC_POWER",
        color="PLANT_ID" if "PLANT_ID" in df.columns else None,
        opacity=0.3,
        color_discrete_sequence=DEFAULT_COLOR_SEQ,
        title="Anomaly Candidates (Low AC Power)",
    )
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["DATE_TIME"],
                y=anomalies["AC_POWER"],
                mode="markers",
                marker=dict(color="red", size=6),
                name="Anomaly",
            )
        )
    return fig


def plot_power_timeseries(df: pd.DataFrame) -> go.Figure:
    """
    Time series of AC and DC power (averaged across inverters).
    """
    df = _ensure_datetime(df)
    required = {"AC_POWER", "DC_POWER"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Missing power columns: {missing}")

    ts_df = (
        df.groupby("DATE_TIME", as_index=False)[["AC_POWER", "DC_POWER"]]
        .mean()
        .melt("DATE_TIME", var_name="Metric", value_name="Power")
    )
    fig = px.line(
        ts_df,
        x="DATE_TIME",
        y="Power",
        color="Metric",
        color_discrete_sequence=DEFAULT_COLOR_SEQ,
        title="Power Time Series",
    )
    return fig


def save_figure(fig: go.Figure, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    return output_path


__all__ = [
    "load_clean_data",
    "plot_daily_energy",
    "plot_inverter_comparison",
    "plot_irradiation_vs_power",
    "plot_temperature_profile",
    "plot_anomaly_points",
    "plot_power_timeseries",
    "save_figure",
]
