import pandas as pd
import streamlit as st

from config import ANOMALY_THRESHOLD
from src.visualization import (
    load_clean_data,
    plot_anomaly_points,
    plot_daily_energy,
    plot_generation_heatmap,
    plot_inverter_comparison,
    plot_irradiation_vs_power,
    plot_power_timeseries,
    plot_temperature_profile,
)


st.set_page_config(
    page_title="Solar Power Analytics",
    layout="wide",
)


@st.cache_data(show_spinner=True)
def get_data(plant: int) -> pd.DataFrame:
    return load_clean_data(plant=plant)


def filter_plant(df: pd.DataFrame, plant_id: str) -> pd.DataFrame:
    if plant_id == "All" or "PLANT_ID" not in df.columns:
        return df
    return df[df["PLANT_ID"] == int(plant_id)].copy()


def build_anomaly_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    if "SOURCE_KEY" not in df.columns and "SOURCE_KEY_gen" in df.columns:
        df["SOURCE_KEY"] = df["SOURCE_KEY_gen"]

    baseline = df.groupby("SOURCE_KEY")["AC_POWER"].transform("mean")
    baseline = baseline.replace(0, pd.NA)
    ratio = df["AC_POWER"] / baseline
    mask = (ratio < threshold).fillna(False)

    anomalies = df.loc[mask, ["DATE_TIME", "SOURCE_KEY", "AC_POWER"]].copy()
    anomalies["MEAN_AC_POWER"] = baseline[mask]
    anomalies["RATIO"] = ratio[mask]
    anomalies = anomalies.sort_values("RATIO", ascending=True)
    return anomalies


def main() -> None:
    st.title("Solar Power Analytics")
    st.caption("Interactive visualization and anomaly exploration on cleaned data.")

    with st.sidebar:
        st.header("Data")
        plant = st.selectbox("Plant", ["All", "1", "2"], index=0)
        st.write("Make sure `data/processed/clean_plant0.csv` exists.")

    df = get_data(plant=0)
    df = filter_plant(df, plant)

    tabs = st.tabs(["Visualization", "Anomaly Detection", "Power Prediction"])

    with tabs[0]:
        st.subheader("Daily Energy")
        st.plotly_chart(plot_daily_energy(df), use_container_width=True)

        st.subheader("Inverter Comparison")
        max_date = df["DATE"].max() if "DATE" in df.columns else None
        selected_date = st.date_input(
            "Comparison date",
            value=max_date,
            max_value=max_date,
        )
        st.plotly_chart(
            plot_inverter_comparison(df, date=selected_date),
            use_container_width=True,
        )

        st.subheader("Irradiation vs Power")
        st.plotly_chart(plot_irradiation_vs_power(df), use_container_width=True)

        st.subheader("Hourly Production Heatmap")
        st.plotly_chart(plot_generation_heatmap(df), use_container_width=True)

        st.subheader("Temperature Profile")
        st.plotly_chart(plot_temperature_profile(df), use_container_width=True)

        st.subheader("Power Time Series")
        st.plotly_chart(plot_power_timeseries(df), use_container_width=True)

    with tabs[1]:
        st.subheader("Anomaly Candidates")
        threshold = st.slider(
            "Threshold (ratio to mean AC power)",
            min_value=0.1,
            max_value=0.9,
            value=float(ANOMALY_THRESHOLD),
            step=0.05,
        )
        st.plotly_chart(
            plot_anomaly_points(df, threshold=threshold),
            use_container_width=True,
        )

        st.subheader("Anomaly List")
        table = build_anomaly_table(df, threshold)
        st.write(f"Rows: {len(table):,}")
        st.dataframe(table, use_container_width=True)

    with tabs[2]:
        st.subheader("Power Prediction")
        st.info(
            "Model integration will be added by the ML team. "
            "This tab will accept irradiation, temperature, and hour inputs."
        )
        st.number_input("Irradiation", value=0.0, step=0.1)
        st.number_input("Module temperature", value=25.0, step=0.1)
        st.number_input("Hour", value=12, min_value=0, max_value=23, step=1)
        st.button("Predict", disabled=True)


if __name__ == "__main__":
    main()
