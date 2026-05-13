import pandas as pd
import streamlit as st

from config import ANOMALY_THRESHOLD
from src.models import predict_power
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

    try:
        df = get_data(plant=0)
    except FileNotFoundError:
        st.error("Temizlenmiş veri bulunamadı. Lütfen önce `python src/data_loader.py` komutunu çalıştırın.")
        return

    df = filter_plant(df, plant)

    tabs = st.tabs(["Visualization", "Anomaly Detection", "Power Prediction"])

    with tabs[0]:
        st.subheader("Daily Energy")
        st.plotly_chart(plot_daily_energy(df), use_container_width=True)

        st.subheader("Inverter Comparison")

        # NaT (Not a Time) hatasını engelleyen güvenli tarih yakalama bloğu
        date_series = None
        if "DATE_TIME" in df.columns:
            valid_dates = pd.to_datetime(df["DATE_TIME"], errors="coerce").dropna()
            if not valid_dates.empty:
                date_series = valid_dates.dt.date

        max_date = date_series.max() if date_series is not None else None

        if max_date is None or pd.isna(max_date):
            selected_date = st.date_input("Comparison date")
        else:
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
        st.subheader("Power Prediction (ML Integration)")
        st.info("Eğitilmiş Random Forest modeli üzerinden anlık enerji üretimi tahmini.")

        col1, col2 = st.columns(2)
        with col1:
            irradiation = st.number_input("Irradiation (W/m2)", value=0.65, step=0.05)
            module_temp = st.number_input("Module temperature (°C)", value=40.0, step=1.0)

            month_names = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
                           7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
            month = st.selectbox("Month", options=list(month_names.keys()), format_func=lambda x: month_names[x],
                                 index=5)

        with col2:
            ambient_temp = st.number_input("Ambient temperature (°C)", value=28.0, step=1.0)
            hour = st.number_input("Hour (0-23)", value=12, min_value=0, max_value=23, step=1)

            day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday",
                         6: "Sunday"}
            day_of_week = st.selectbox("Day of Week", options=list(day_names.keys()),
                                       format_func=lambda x: day_names[x], index=0)

        plant_label = "Combined" if plant == "All" else f"Plant_{plant}"

        if st.button("Predict", type="primary"):
            try:
                prediction = predict_power(
                    irradiation=irradiation,
                    ambient_temperature=ambient_temp,
                    module_temperature=module_temp,
                    hour=hour,
                    day_of_week=day_of_week,
                    month=month,
                    plant_label=plant_label,
                    model_name="random_forest"
                )
                st.success(f"Tahmin Edilen AC Gücü: **{prediction:.2f} W**")
            except FileNotFoundError:
                st.error(
                    f"Model ({plant_label}) bulunamadı! Lütfen önce `python src/models.py` komutunu çalıştırarak modelleri eğitin.")
            except Exception as e:
                st.error(f"Tahmin işlemi sırasında bir hata oluştu: {e}")


if __name__ == "__main__":
    main()