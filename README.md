# solar-power-analytics

## Takim

- Ali İhsan Çevik
- Furkan Kopan
- Mesut Altun
- Cem Akan
- Veysel Genç
- Yunus Emre Erten
- Zeliha İnan
- Gülse Ogultegin
- Sinem Durmaz
- Berat Erhan Şekeröz (Temsilci)
- Zeynep Elif Göksu
- Ebrar Kalfaoğlu
- Miray Balıkoğlu
- Arda Yenisaraç
- Kerem Nalçabasmaz

## Veri kaynagi

- Dataset: Solar Power Generation Data
- URL: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

### Veri indirip yerlestirme

**Secenek 1: KaggleHub ile (onerilen)**

```zsh
python -m pip install kagglehub
python - <<'PY'
import shutil
import os
import kagglehub

path = kagglehub.dataset_download("anikannal/solar-power-generation-data")
raw_dir = os.path.join(os.getcwd(), "data", "raw")
os.makedirs(raw_dir, exist_ok=True)

for name in [
    "Plant_1_Generation_Data.csv",
    "Plant_1_Weather_Sensor_Data.csv",
    "Plant_2_Generation_Data.csv",
    "Plant_2_Weather_Sensor_Data.csv",
]:
    shutil.copy(os.path.join(path, name), os.path.join(raw_dir, name))

print("Copied files to", raw_dir)
PY
```

**Secenek 2: Manuel indirme**

1) Kaggle’dan zip indir.
2) Asagidaki 4 CSV dosyasini `data/raw/` klasorune koy:
   - `Plant_1_Generation_Data.csv`
   - `Plant_1_Weather_Sensor_Data.csv`
   - `Plant_2_Generation_Data.csv`
   - `Plant_2_Weather_Sensor_Data.csv`

Sonra temiz veri uretmek icin:

```zsh
python src/data_loader.py
```

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
