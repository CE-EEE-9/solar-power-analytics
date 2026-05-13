import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    PLANT1_GENERATION,
    PLANT1_WEATHER,
    PLANT2_GENERATION,
    PLANT2_WEATHER,
    GECE_BASLANGIC,
    GECE_BITIS,
)


def load_raw_data():
    """
    Kaggle'dan indirilen ham CSV dosyalarını okur.
    """

    plant1_gen = pd.read_csv(os.path.join(RAW_DATA_PATH, PLANT1_GENERATION))
    plant1_weather = pd.read_csv(os.path.join(RAW_DATA_PATH, PLANT1_WEATHER))

    plant2_gen = pd.read_csv(os.path.join(RAW_DATA_PATH, PLANT2_GENERATION))
    plant2_weather = pd.read_csv(os.path.join(RAW_DATA_PATH, PLANT2_WEATHER))

    print("Ham veriler yüklendi.")
    print("Plant 1 Generation:", plant1_gen.shape)
    print("Plant 1 Weather:", plant1_weather.shape)
    print("Plant 2 Generation:", plant2_gen.shape)
    print("Plant 2 Weather:", plant2_weather.shape)

    return plant1_gen, plant1_weather, plant2_gen, plant2_weather


def fix_datetime(df):
    """
    DATE_TIME sütununu uyarı vermeden datetime formatına çevirir.
    Dataset içinde farklı tarih formatları olduğu için formatları sırayla dener.
    """

    df = df.copy()

    date_strings = df["DATE_TIME"].astype(str)

    # Format 1: 15-05-2020 00:00
    parsed_dates = pd.to_datetime(
        date_strings,
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    # Format 2: 2020-05-15 00:00:00
    missing_mask = parsed_dates.isna()
    parsed_dates.loc[missing_mask] = pd.to_datetime(
        date_strings.loc[missing_mask],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    # Format 3: 2020-05-15 00:00
    missing_mask = parsed_dates.isna()
    parsed_dates.loc[missing_mask] = pd.to_datetime(
        date_strings.loc[missing_mask],
        format="%Y-%m-%d %H:%M",
        errors="coerce"
    )

    df["DATE_TIME"] = parsed_dates

    df = df.dropna(subset=["DATE_TIME"])

    df["DATE"] = df["DATE_TIME"].dt.date
    df["HOUR"] = df["DATE_TIME"].dt.hour
    df["DAY_OF_WEEK"] = df["DATE_TIME"].dt.dayofweek
    df["MONTH"] = df["DATE_TIME"].dt.month

    return df


def clean_generation_data(df):
    """
    Üretim verisini temizler.
    Negatif değerleri sıfırlar.
    Gece saatlerinde üretimi 0 kabul eder.
    """

    df = df.copy()

    df = fix_datetime(df)

    df = df.sort_values("DATE_TIME")
    df = df.ffill()
    df = df.dropna()

    df["DC_POWER"] = df["DC_POWER"].clip(lower=0)
    df["AC_POWER"] = df["AC_POWER"].clip(lower=0)

    night_mask = (df["HOUR"] < GECE_BASLANGIC) | (df["HOUR"] > GECE_BITIS)

    df.loc[night_mask, "DC_POWER"] = 0
    df.loc[night_mask, "AC_POWER"] = 0

    return df


def clean_weather_data(df):
    """
    Hava/sensör verisini temizler.
    Negatif irradiation değerlerini sıfırlar.
    """

    df = df.copy()

    df = fix_datetime(df)

    df = df.sort_values("DATE_TIME")
    df = df.ffill()
    df = df.dropna()

    if "IRRADIATION" in df.columns:
        df["IRRADIATION"] = df["IRRADIATION"].clip(lower=0)

    return df


def merge_data(generation_df, weather_df):
    """
    Üretim verisi ile hava verisini DATE_TIME ve PLANT_ID üzerinden birleştirir.
    """

    merged = pd.merge(
        generation_df,
        weather_df,
        on=["DATE_TIME", "PLANT_ID"],
        how="inner",
        suffixes=("_gen", "_weather")
    )

    drop_columns = [
        "DATE_weather",
        "HOUR_weather",
        "DAY_OF_WEEK_weather",
        "MONTH_weather",
        "DATE_gen",
    ]

    for col in drop_columns:
        if col in merged.columns:
            merged = merged.drop(columns=col)

    rename_map = {
        "HOUR_gen": "HOUR",
        "DAY_OF_WEEK_gen": "DAY_OF_WEEK",
        "MONTH_gen": "MONTH",
    }

    merged = merged.rename(columns=rename_map)

    return merged


def load_data(plant=0):
    """
    plant=1: Plant 1 verisi
    plant=2: Plant 2 verisi
    plant=0: Plant 1 + Plant 2 birleşik veri
    """

    plant1_gen, plant1_weather, plant2_gen, plant2_weather = load_raw_data()

    print("\nPlant 1 işleniyor...")
    plant1_gen = clean_generation_data(plant1_gen)
    plant1_weather = clean_weather_data(plant1_weather)
    merged1 = merge_data(plant1_gen, plant1_weather)

    print("\nPlant 2 işleniyor...")
    plant2_gen = clean_generation_data(plant2_gen)
    plant2_weather = clean_weather_data(plant2_weather)
    merged2 = merge_data(plant2_gen, plant2_weather)

    if plant == 1:
        df = merged1
    elif plant == 2:
        df = merged2
    else:
        df = pd.concat([merged1, merged2], ignore_index=True)

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    save_path = os.path.join(PROCESSED_DATA_PATH, f"clean_plant{plant}.csv")
    df.to_csv(save_path, index=False)

    print("\nTemizlenmiş veri kaydedildi:", save_path)
    print("Veri boyutu:", df.shape)
    print("Sütunlar:", list(df.columns))

    return df


if __name__ == "__main__":
    df = load_data(plant=0)
    print("\nİlk 5 satır:")
    print(df.head())