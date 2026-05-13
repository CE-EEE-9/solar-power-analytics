import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

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

    # Sadece güneş ışınımı olan kayıtlar kullanılıyor.
    # Böylece model gece saatlerindeki direkt 0 üretimden öğrenmiyor.
    df = df[df["IRRADIATION"] > 0].copy()

    df = df.dropna(subset=["AC_POWER"])

    feature_columns = [
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION",
        "HOUR",
        "DAY_OF_WEEK",
        "MONTH",
    ]

    df = df.dropna(subset=feature_columns)

    X = df[feature_columns]
    y = df["AC_POWER"]

    return X, y, feature_columns


def evaluate_regression_model(model, X_train, X_test, y_train, y_test):
    """
    Modeli eğitir ve MAE, RMSE, R2 metriklerini hesaplar.
    """

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return mae, rmse, r2, y_pred


def create_actual_vs_predicted_plot(y_test, y_pred, model_name, plant_label):
    """
    Gerçek AC_POWER ve tahmin edilen AC_POWER grafiğini kaydeder.
    """

    os.makedirs(REPORTS_PATH, exist_ok=True)

    plt.figure(figsize=(9, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual AC Power")
    plt.ylabel("Predicted AC Power")
    plt.title(f"Actual vs Predicted AC Power - {model_name} - {plant_label}")
    plt.tight_layout()

    file_name = f"ml_actual_vs_predicted_{plant_label}.png"
    save_path = os.path.join(REPORTS_PATH, file_name)

    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Actual vs Predicted grafiği kaydedildi:", save_path)


def create_feature_importance_plot(model, feature_columns, plant_label):
    """
    Random Forest için feature importance grafiğini kaydeder.
    """

    os.makedirs(REPORTS_PATH, exist_ok=True)

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance:")
    print(importance_df)

    plt.figure(figsize=(9, 6))
    plt.bar(importance_df["Feature"], importance_df["Importance"])
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title(f"Feature Importance - Random Forest - {plant_label}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    file_name = f"ml_feature_importance_{plant_label}.png"
    save_path = os.path.join(REPORTS_PATH, file_name)

    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Feature importance grafiği kaydedildi:", save_path)

    return importance_df


def train_models_for_plant(plant, plant_label):
    """
    Belirli bir plant için Linear Regression, Decision Tree ve Random Forest eğitir.
    """

    print("\n" + "=" * 70)
    print(f"{plant_label} için model eğitimi başlıyor")
    print("=" * 70)

    print("\nVeri yükleniyor...")
    df = load_data(plant=plant)

    print("\nML verisi hazırlanıyor...")
    X, y, feature_columns = prepare_ml_data(df)

    print("Feature sayısı:", X.shape[1])
    print("Satır sayısı:", X.shape[0])
    print("Target:", "AC_POWER")
    print("Kullanılan feature'lar:", feature_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            random_state=RANDOM_STATE,
            max_depth=RF_MAX_DEPTH
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    results = []
    predictions = {}

    for model_name, model in models.items():
        print("\nModel eğitiliyor:", model_name)

        mae, rmse, r2, y_pred = evaluate_regression_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append({
            "Dataset": plant_label,
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2
        })

        predictions[model_name] = y_pred

        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("R2 Score:", round(r2, 4))

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="R2 Score", ascending=False)

    print(f"\n{plant_label} Model Karşılaştırma:")
    print(results_df)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    best_prediction = predictions[best_model_name]

    print(f"\n{plant_label} için en iyi model:", best_model_name)

    create_actual_vs_predicted_plot(
        y_test,
        best_prediction,
        best_model_name,
        plant_label
    )

    # Feature importance sadece Random Forest için anlamlıdır.
    random_forest = models["Random Forest"]
    create_feature_importance_plot(
        random_forest,
        feature_columns,
        plant_label
    )

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    model_file_name = f"best_solar_power_model_{plant_label}.pkl"
    model_path = os.path.join(MODEL_SAVE_PATH, model_file_name)

    joblib.dump(best_model, model_path)

    print("En iyi model kaydedildi:", model_path)

    return results_df


def train_all_models():
    """
    Plant 1, Plant 2 ve birleşik veri için modelleri çalıştırır.
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


if __name__ == "__main__":
    train_all_models()