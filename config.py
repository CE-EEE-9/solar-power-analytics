import os

# ================================
# DOSYA YOLLARI
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models")

# ================================
# VERİ DOSYASI İSİMLERİ
# ================================
PLANT1_GENERATION = "Plant_1_Generation_Data.csv"
PLANT1_WEATHER = "Plant_1_Weather_Sensor_Data.csv"
PLANT2_GENERATION = "Plant_2_Generation_Data.csv"
PLANT2_WEATHER = "Plant_2_Weather_Sensor_Data.csv"

# ================================
# VERİ ÖN İŞLEME PARAMETRELERİ
# ================================
DATETIME_FORMAT = "%d-%m-%Y %H:%M"
GECE_BASLANGIC = 6     # 06:00'dan önce üretim olmaz
GECE_BITIS = 18        # 18:00'dan sonra üretim olmaz

# ================================
# MODEL PARAMETRELERİ
# ================================
TEST_SIZE = 0.2
RANDOM_STATE = 42
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10

# ================================
# ANOMALİ TESPİT EŞİĞİ
# ================================
ANOMALY_THRESHOLD = 0.80  # Ortalamadan %20 düşük olanlar anomali