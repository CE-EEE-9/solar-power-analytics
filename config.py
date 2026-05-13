import os

# ================================
# DOSYA YOLLARI
# ================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models")
REPORTS_PATH = os.path.join(BASE_DIR, "reports")

# ================================
# VERİ DOSYASI İSİMLERİ
# ================================

PLANT1_GENERATION = "Plant_1_Generation_Data.csv"
PLANT1_WEATHER = "Plant_1_Weather_Sensor_Data.csv"
PLANT2_GENERATION = "Plant_2_Generation_Data.csv"
PLANT2_WEATHER = "Plant_2_Weather_Sensor_Data.csv"

# ================================
# MODEL PARAMETRELERİ
# ================================

TEST_SIZE = 0.2
RANDOM_STATE = 42
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10

# ================================
# VERİ TEMİZLEME PARAMETRELERİ
# ================================

GECE_BASLANGIC = 6
GECE_BITIS = 18

# ================================
# ANOMALY DETECTION
# ================================

ANOMALY_THRESHOLD = 0.80  # 20% below mean
