import argparse
import os
import sys
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.visualization import (
    load_clean_data,
    plot_anomaly_points,
    plot_daily_energy,
    plot_inverter_comparison,
    plot_irradiation_vs_power,
    plot_power_timeseries,
    plot_temperature_profile,
)


def build_figures(df):
    return {
        "daily_energy": plot_daily_energy(df),
        "inverter_comparison": plot_inverter_comparison(df),
        "irradiation_vs_power": plot_irradiation_vs_power(df),
        "power_timeseries": plot_power_timeseries(df),
        "temperature_profile": plot_temperature_profile(df),
        "anomaly_points": plot_anomaly_points(df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Save visualization PNG files.")
    parser.add_argument("--plant", type=int, default=0, help="Plant id (0=all)")
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT_DIR, "reports", "images"),
        help="Output directory for PNG files",
    )
    args = parser.parse_args()

    df = load_clean_data(plant=args.plant)
    figures = build_figures(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output, exist_ok=True)

    try:
        for name, fig in figures.items():
            filename = f"viz_{name}_plant{args.plant}_{timestamp}.png"
            fig.write_image(os.path.join(args.output, filename))
    except Exception as exc:
        raise SystemExit(
            "Image export needs kaleido. Install it with: "
            "pip install -r requirements-images.txt"
        ) from exc

    print(f"Saved {len(figures)} charts to {args.output}")


if __name__ == "__main__":
    main()

