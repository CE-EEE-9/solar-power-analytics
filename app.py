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

    # Kaggle veri setindeki gerçek kimlik numaraları ile eşleştirme yapıyoruz.
    actual_plant_id = 4135001 if plant_id == "1" else 4136001

    return df[df["PLANT_ID"] == actual_plant_id].copy()


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


def get_model_plant_label(selected_plant: str) -> str:
    if selected_plant == "1":
        return "Plant_1"
    if selected_plant == "2":
        return "Plant_2"
    return "Combined"


def load_model_results() -> pd.DataFrame | None:
    try:
        results_df = pd.read_csv("reports/ml_all_model_results.csv")

        # Tabloyu daha okunabilir göstermek için yuvarlama.
        numeric_columns = ["MAE", "RMSE", "R2 Score"]

        for column in numeric_columns:
            if column in results_df.columns:
                results_df[column] = results_df[column].round(4)

        return results_df

    except FileNotFoundError:
        return None


def main() -> None:
    st.title("Solar Power Analytics")
    st.caption(
        "Interactive visualization, anomaly detection, and ML-based power prediction."
    )

    with st.sidebar:
        st.header("Data")
        plant = st.selectbox("Plant", ["All", "1", "2"], index=0)
        st.write("Make sure `data/processed/clean_plant0.csv` exists.")

    try:
        df = get_data(plant=0)
    except FileNotFoundError:
        st.error(
            "Cleaned data was not found. Please run "
            "`python src/data_loader.py` first."
        )
        return

    df = filter_plant(df, plant)

    tabs = st.tabs(["Visualization", "Anomaly Detection", "Power Prediction"])

    with tabs[0]:
        st.subheader("Daily Energy")
        st.plotly_chart(plot_daily_energy(df), use_container_width=True)

        st.subheader("Inverter Comparison")

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
        st.subheader("Power Prediction")

        st.write(
            "Enter weather and time values to estimate AC power generation "
            "using the trained machine learning models."
        )

        col1, col2 = st.columns(2)

        with col1:
            selected_model = st.selectbox(
                "Model",
                ["random_forest", "linear_regression"],
                index=0,
            )

            irradiation = st.number_input(
                "Irradiation",
                min_value=0.0,
                max_value=2.0,
                value=0.65,
                step=0.01,
            )

            ambient_temperature = st.number_input(
                "Ambient temperature",
                min_value=-10.0,
                max_value=60.0,
                value=28.0,
                step=0.1,
            )

            month_names = {
                1: "January",
                2: "February",
                3: "March",
                4: "April",
                5: "May",
                6: "June",
                7: "July",
                8: "August",
                9: "September",
                10: "October",
                11: "November",
                12: "December",
            }

            month = st.selectbox(
                "Month",
                options=list(month_names.keys()),
                format_func=lambda x: month_names[x],
                index=5,
            )

        with col2:
            module_temperature = st.number_input(
                "Module temperature",
                min_value=-10.0,
                max_value=90.0,
                value=40.0,
                step=0.1,
            )

            hour = st.number_input(
                "Hour",
                min_value=0,
                max_value=23,
                value=12,
                step=1,
            )

            day_names = {
                0: "Monday",
                1: "Tuesday",
                2: "Wednesday",
                3: "Thursday",
                4: "Friday",
                5: "Saturday",
                6: "Sunday",
            }

            day_of_week = st.selectbox(
                "Day of week",
                options=list(day_names.keys()),
                format_func=lambda x: day_names[x],
                index=2,
            )

        plant_label = get_model_plant_label(plant)

        st.info(f"Selected prediction dataset: `{plant_label}`")

        if st.button("Predict AC Power", type="primary"):
            try:
                prediction = predict_power(
                    irradiation=irradiation,
                    ambient_temperature=ambient_temperature,
                    module_temperature=module_temperature,
                    hour=hour,
                    day_of_week=day_of_week,
                    month=month,
                    plant_label=plant_label,
                    model_name=selected_model,
                )

                st.success(f"Predicted AC Power: **{prediction:.2f} W**")

            except FileNotFoundError as error:
                st.error(str(error))
                st.warning(
                    "If the model file is missing, run this command first: "
                    "`python src/models.py`"
                )

            except Exception as error:
                st.error("Prediction failed.")
                st.exception(error)

        st.divider()

        st.subheader("Model Performance Results")

        results_df = load_model_results()

        if results_df is None:
            st.warning(
                "Model performance results were not found. "
                "Run `python src/models.py` to generate "
                "`reports/ml_all_model_results.csv`."
            )
        else:
            st.write(
                "The table below shows MAE, RMSE, and R² scores for "
                "Linear Regression and Random Forest models."
            )

            if plant == "All":
                filtered_results = results_df
            else:
                selected_dataset = get_model_plant_label(plant)
                filtered_results = results_df[
                    results_df["Dataset"] == selected_dataset
                ]

            st.dataframe(filtered_results, use_container_width=True)


if __name__ == "__main__":
    main()