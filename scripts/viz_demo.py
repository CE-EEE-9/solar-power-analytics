import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.visualization import (
    plot_anomaly_points,
    plot_daily_energy,
    plot_inverter_comparison,
    plot_irradiation_vs_power,
    plot_power_timeseries,
    plot_temperature_profile,
    save_figure,
)


def _make_synthetic_data(days: int = 7, freq_minutes: int = 15) -> pd.DataFrame:
    start = datetime(2020, 5, 1)
    periods = int((24 * 60 / freq_minutes) * days)
    dt_index = pd.date_range(start, periods=periods, freq=f"{freq_minutes}min")

    plant_ids = [1, 2]
    inverters_per_plant = 3

    rows = []
    for plant_id in plant_ids:
        for inv_idx in range(inverters_per_plant):
            source_key = f"INV_{plant_id}_{inv_idx + 1}"
            daily_yield = 0.0
            for ts in dt_index:
                hour = ts.hour + ts.minute / 60
                irradiation = max(0.0, np.sin((hour - 6) / 12 * np.pi))
                irradiation *= 900 + np.random.normal(0, 30)
                irradiation = max(0.0, irradiation)

                dc_power = irradiation * (0.6 + 0.05 * inv_idx) + np.random.normal(0, 20)
                ac_power = max(0.0, dc_power * 0.85 + np.random.normal(0, 15))

                if ts.hour == 0 and ts.minute == 0:
                    daily_yield = 0.0
                daily_yield += ac_power * (freq_minutes / 60)

                total_yield = daily_yield + plant_id * 10000 + inv_idx * 500

                ambient = 20 + 10 * np.sin((hour - 6) / 12 * np.pi) + np.random.normal(0, 1)
                module = ambient + 5 + irradiation / 300 + np.random.normal(0, 1)

                rows.append(
                    {
                        "DATE_TIME": ts,
                        "PLANT_ID": plant_id,
                        "SOURCE_KEY": source_key,
                        "DC_POWER": max(0.0, dc_power),
                        "AC_POWER": max(0.0, ac_power),
                        "DAILY_YIELD": daily_yield,
                        "TOTAL_YIELD": total_yield,
                        "AMBIENT_TEMPERATURE": ambient,
                        "MODULE_TEMPERATURE": module,
                        "IRRADIATION": irradiation,
                        "DATE": ts.date(),
                        "HOUR": ts.hour,
                        "DAY_OF_WEEK": ts.dayofweek,
                        "MONTH": ts.month,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    df = _make_synthetic_data()

    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    outputs = {
        "daily_energy": plot_daily_energy(df),
        "inverter_comparison": plot_inverter_comparison(df),
        "irradiation_vs_power": plot_irradiation_vs_power(df),
        "power_timeseries": plot_power_timeseries(df),
        "temperature_profile": plot_temperature_profile(df),
        "anomaly_points": plot_anomaly_points(df),
    }

    for name, fig in outputs.items():
        save_figure(fig, os.path.join(reports_dir, f"viz_demo_{name}.html"))

    print(f"Saved {len(outputs)} demo charts to {reports_dir}")


if __name__ == "__main__":
    main()
