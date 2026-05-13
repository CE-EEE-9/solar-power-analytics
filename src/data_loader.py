import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (RAW_DATA_PATH, PROCESSED_DATA_PATH,
                    PLANT1_GENERATION, PLANT1_WEATHER,
                    PLANT2_GENERATION, PLANT2_WEATHER,
                    DATETIME_FORMAT, GECE_BASLANGIC, GECE_BITIS)


def load_raw_data():
    """
    Ham CSV dosyalarını okur ve döndürür.
    """
    plant1_gen = pd.read_csv(os.path.join(RAW_DATA_PATH, PLANT1_GENERATION))
    plant1_weather = pd.read_csv(os.path.join(RAW_DATA_PATH, PLANT1_WEATHER))
    plant2_gen = pd.read_csv(os.path.join(RAW_DATA_PATH, PLANT2_GENERATION))
    plant2_weather = pd.read_csv(os.path.join(RAW_DATA_PATH, PLANT2_WEATHER))

    print("✅ Ham veriler yüklendi.")
    print(f"   Plant 1 Üretim: {plant1_gen.shape}")
    print(f"   Plant 1 Hava  : {plant1_weather.shape}")
    print(f"   Plant 2 Üretim: {plant2_gen.shape}")
    print(f"   Plant 2 Hava  : {plant2_weather.shape}")

    return plant1_gen, plant1_weather, plant2_gen, plant2_weather


def fix_datetime(df, filename=""):
    """
    Tarih sütununu düzenler. Plant1_Generation farklı formatta olduğu için
    ayrı ele alınır.
    """
    df = df.copy()
    
    # Plant1_Generation farklı format kullanıyor
    if "Plant_1_Generation" in filename:
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], 
                                          format="%d-%m-%Y %H:%M")
    else:
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], 
                                          format="%Y-%m-%d %H:%M:%S")
    
    df['DATE'] = df['DATE_TIME'].dt.date
    df['HOUR'] = df['DATE_TIME'].dt.hour
    df['DAY_OF_WEEK'] = df['DATE_TIME'].dt.dayofweek
    df['MONTH'] = df['DATE_TIME'].dt.month
    return df


def clean_data(df, is_generation=True):
    """
    Eksik değerleri ve aykırı değerleri temizler.
    """
    df = df.copy()

    # Eksik değerleri temizle
    missing_before = df.isnull().sum().sum()
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    missing_after = df.isnull().sum().sum()
    print(f"   Eksik değer: {missing_before} → {missing_after}")

    # Üretim verisi için gece saatlerini sıfırla
    if is_generation:
        gece_mask = (df['HOUR'] < GECE_BASLANGIC) | (df['HOUR'] > GECE_BITIS)
        df.loc[gece_mask, 'DC_POWER'] = 0
        df.loc[gece_mask, 'AC_POWER'] = 0

        # Negatif değerleri sıfırla
        df['DC_POWER'] = df['DC_POWER'].clip(lower=0)
        df['AC_POWER'] = df['AC_POWER'].clip(lower=0)

    return df


def merge_data(gen_df, weather_df):
    """
    Üretim verisi ile hava sensörü verisini birleştirir.
    """
    merged = pd.merge(gen_df, weather_df,
                      on=['DATE_TIME', 'PLANT_ID'],
                      how='inner',
                      suffixes=('_gen', '_weather'))

    # Duplicate sütunları temizle, weather'dan gelenleri at
    cols_to_drop = ['DATE_weather', 'HOUR_weather', 
                    'DAY_OF_WEEK_weather', 'MONTH_weather',
                    'DATE_gen']
    merged.drop(columns=cols_to_drop, inplace=True)

    # Sütun isimlerini temizle
    merged.rename(columns={
        'HOUR_gen': 'HOUR',
        'DAY_OF_WEEK_gen': 'DAY_OF_WEEK',
        'MONTH_gen': 'MONTH'
    }, inplace=True)

    print(f"   Birleştirme sonrası: {merged.shape}")
    print(f"   Sütunlar: {list(merged.columns)}")
    return merged


def load_data(plant=1):
    """
    Ana fonksiyon. Ham veriyi yükler, temizler, birleştirir ve kaydeder.
    plant=1 → Plant 1 verisi
    plant=2 → Plant 2 verisi
    plant=0 → Her iki santralin verisi birleşik
    """
    plant1_gen, plant1_weather, plant2_gen, plant2_weather = load_raw_data()

    print("\n🔧 Plant 1 işleniyor...")
    plant1_gen = fix_datetime(plant1_gen, filename="Plant_1_Generation")
    plant1_weather = fix_datetime(plant1_weather)
    plant1_gen = clean_data(plant1_gen, is_generation=True)
    plant1_weather = clean_data(plant1_weather, is_generation=False)
    merged1 = merge_data(plant1_gen, plant1_weather)

    print("\n🔧 Plant 2 işleniyor...")
    plant2_gen = fix_datetime(plant2_gen)
    plant2_weather = fix_datetime(plant2_weather)
    plant2_gen = clean_data(plant2_gen, is_generation=True)
    plant2_weather = clean_data(plant2_weather, is_generation=False)
    merged2 = merge_data(plant2_gen, plant2_weather)

    # Seçime göre döndür
    if plant == 1:
        df = merged1
    elif plant == 2:
        df = merged2
    else:
        df = pd.concat([merged1, merged2], ignore_index=True)

    # Temizlenmiş veriyi kaydet
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    save_path = os.path.join(PROCESSED_DATA_PATH, f"clean_plant{plant}.csv")
    df.to_csv(save_path, index=False)
    print(f"\n✅ Temizlenmiş veri kaydedildi: {save_path}")
    print(f"   Toplam satır: {len(df)}")
    print(f"   Toplam sütun: {len(df.columns)}")
    print(f"   Sütunlar: {list(df.columns)}")

    return df


if __name__ == "__main__":
    df = load_data(plant=0)
    print("\n📊 İlk 5 satır:")
    print(df.head())