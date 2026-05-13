import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_SAVE_PATH,
    REPORTS_PATH,
    RANDOM_STATE,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    TEST_SIZE,
)
from src.data_loader import load_data


FEATURE_COLUMNS = [
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "HOUR",
    "DAY_OF_WEEK",
    "MONTH",
]


def prepare_ml_data(df):
    """
    Makine öğrenmesi için veriyi hazırlar.
    Hedef değişken: AC_POWER
    """

    df = df.copy()

    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], errors="coerce")
    df = df.dropna(subset=["DATE_TIME"])

    if "HOUR" not in df.columns:
        df["HOUR"] = df["DATE_TIME"].dt.hour

    if "DAY_OF_WEEK" not in df.columns:
        df["DAY_OF_WEEK"] = df["DATE_TIME"].dt.dayofweek

    if "MONTH" not in df.columns:
        df["MONTH"] = df["DATE_TIME"].dt.month

    # Gece saatlerindeki direkt sıfır üretim etkisini azaltmak için
    # yalnızca ışınım olan kayıtları kullanıyoruz.
    df = df[df["IRRADIATION"] > 0].copy()

    df = df.dropna(subset=["AC_POWER"])
    df = df.dropna(subset=FEATURE_COLUMNS)

    X = df[FEATURE_COLUMNS]
    y = df["AC_POWER"]

    return X, y


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Modeli eğitir ve performans metriklerini hesaplar.
    """

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return mae, rmse, r2, y_pred


def save_model(model, model_name, plant_label):
    """
    Eğitilen modeli saved_models klasörüne kaydeder.
    """

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    file_name = f"{model_name}_{plant_label}.pkl"
    save_path = os.path.join(MODEL_SAVE_PATH, file_name)

    joblib.dump(model, save_path)

    print("Model kaydedildi:", save_path)


def create_actual_vs_predicted_plot(y_test, y_pred, model_name, plant_label):
    """
    Gerçek ve tahmin edilen AC_POWER değerlerini grafik olarak kaydeder.
    """

    os.makedirs(REPORTS_PATH, exist_ok=True)

    plt.figure(figsize=(9, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual AC Power")
    plt.ylabel("Predicted AC Power")
    plt.title(f"Actual vs Predicted AC Power - {model_name} - {plant_label}")
    plt.tight_layout()

    save_path = os.path.join(
        REPORTS_PATH,
        f"ml_actual_vs_predicted_{model_name}_{plant_label}.png"
    )

    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Actual vs Predicted grafiği kaydedildi:", save_path)


def create_feature_importance_plot(model, plant_label):
    """
    Random Forest modelinin feature importance grafiğini oluşturur.
    """

    os.makedirs(REPORTS_PATH, exist_ok=True)

    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance:")
    print(importance_df)

    importance_csv_path = os.path.join(
        REPORTS_PATH,
        f"ml_feature_importance_{plant_label}.csv"
    )
    importance_df.to_csv(importance_csv_path, index=False)

    plt.figure(figsize=(9, 6))
    plt.bar(importance_df["Feature"], importance_df["Importance"])
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title(f"Feature Importance - Random Forest - {plant_label}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_path = os.path.join(
        REPORTS_PATH,
        f"ml_feature_importance_{plant_label}.png"
    )

    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Feature importance CSV kaydedildi:", importance_csv_path)
    print("Feature importance grafiği kaydedildi:", save_path)

    return importance_df


def train_models_for_plant(plant, plant_label):
    """
    Belirli bir veri seti için Linear Regression ve Random Forest modellerini eğitir.
    """

    print("\n" + "=" * 70)
    print(f"{plant_label} için model eğitimi başlıyor")
    print("=" * 70)

    print("\nVeri yükleniyor...")
    df = load_data(plant=plant)

    print("\nML verisi hazırlanıyor...")
    X, y = prepare_ml_data(df)

    print("Feature sayısı:", X.shape[1])
    print("Satır sayısı:", X.shape[0])
    print("Target:", "AC_POWER")
    print("Kullanılan feature'lar:", FEATURE_COLUMNS)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    result_rows = []
    trained_models = {}

    for model_name, model in models.items():
        print("\nModel eğitiliyor:", model_name)

        mae, rmse, r2, y_pred = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        result_rows.append({
            "Dataset": plant_label,
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2
        })

        trained_models[model_name] = model

        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("R2 Score:", round(r2, 4))

        save_model(model, model_name, plant_label)

        create_actual_vs_predicted_plot(
            y_test,
            y_pred,
            model_name,
            plant_label
        )

    # Feature importance Random Forest üzerinden çıkarılır.
    create_feature_importance_plot(
        trained_models["random_forest"],
        plant_label
    )

    results_df = pd.DataFrame(result_rows)
    results_df = results_df.sort_values(by="R2 Score", ascending=False)

    print(f"\n{plant_label} Model Karşılaştırma:")
    print(results_df)

    return results_df


def train_all_models():
    """
    Plant 1, Plant 2 ve birleşik veri için modelleri eğitir.
    """

    all_results = []

    plant_configs = [
        (1, "Plant_1"),
        (2, "Plant_2"),
        (0, "Combined"),
    ]

    for plant, plant_label in plant_configs:
        result_df = train_models_for_plant(plant, plant_label)
        all_results.append(result_df)

    final_results = pd.concat(all_results, ignore_index=True)

    final_results = final_results.sort_values(
        by=["Dataset", "R2 Score"],
        ascending=[True, False]
    )

    os.makedirs(REPORTS_PATH, exist_ok=True)

    final_results_path = os.path.join(
        REPORTS_PATH,
        "ml_all_model_results.csv"
    )

    final_results.to_csv(final_results_path, index=False)

    print("\n" + "=" * 70)
    print("TÜM MODEL SONUÇLARI")
    print("=" * 70)
    print(final_results)

    print("\nTüm model sonuçları kaydedildi:", final_results_path)

    return final_results


def predict_power(
    irradiation,
    ambient_temperature,
    module_temperature,
    hour,
    day_of_week=0,
    month=6,
    plant_label="Combined",
    model_name="random_forest"
):
    """
    Dışarıdan verilen ışınım, sıcaklık ve zaman değerlerine göre
    AC_POWER tahmini yapar.

    Örnek:
    predict_power(
        irradiation=0.65,
        ambient_temperature=28,
        module_temperature=40,
        hour=12
    )
    """

    model_path = os.path.join(
        MODEL_SAVE_PATH,
        f"{model_name}_{plant_label}.pkl"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model bulunamadı: {model_path}. "
            "Önce python src/models.py komutunu çalıştırarak modelleri eğitin."
        )

    model = joblib.load(model_path)

    input_data = pd.DataFrame([{
        "AMBIENT_TEMPERATURE": ambient_temperature,
        "MODULE_TEMPERATURE": module_temperature,
        "IRRADIATION": irradiation,
        "HOUR": hour,
        "DAY_OF_WEEK": day_of_week,
        "MONTH": month,
    }])

    prediction = model.predict(input_data)[0]

    return prediction


if __name__ == "__main__":
    train_all_models()

    print("\n" + "=" * 70)
    print("ÖRNEK TAHMİN")
    print("=" * 70)

    sample_prediction = predict_power(
        irradiation=0.65,
        ambient_temperature=28,
        module_temperature=40,
        hour=12,
        day_of_week=2,
        month=6,
        plant_label="Combined",
        model_name="random_forest"
    )

    print("Tahmin edilen AC_POWER:", round(sample_prediction, 2), "W")