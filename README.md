# solar-power-analytics

## Visualization quick start

1) Generate cleaned data (once):
   - Run `src/data_loader.py` to create `data/processed/clean_plant0.csv`.
2) Build plots in `src/visualization.py` and call the functions from `app.py`.
3) Optional: run the demo script to verify charts without raw data.

## Demo charts (synthetic data)

Run the demo script to create HTML charts in `reports/`:

```zsh
python scripts/viz_demo.py
```

## Save all charts from cleaned data (HTML)

Create HTML reports using the real cleaned dataset:

```zsh
python scripts/save_visuals.py --plant 0
```

Files are saved to `reports/` with a timestamp. Open any HTML file in your browser:

```zsh
open reports/viz_daily_energy_plant0_YYYYMMDD_HHMMSS.html
```

## Save all charts as images (PNG)

Install the image backend once:

```zsh
python -m pip install -r requirements-images.txt
```

Then export PNG files:

```zsh
python scripts/save_images.py --plant 0
```

PNG files are saved under `reports/images/`.

## Notes

- Cleaned CSV files are ignored by git; everyone generates them locally.
- `requirements.txt` already includes plotly, pandas, numpy, and streamlit.
