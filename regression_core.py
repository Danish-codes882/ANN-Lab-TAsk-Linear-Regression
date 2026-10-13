"""Shared functions for training and using the two Linear Regression models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "energy": {
        "title": "Building Energy Consumption",
        "file": ROOT / "datasets" / "urban_building_energy.csv",
        "target": "monthly_energy_kwh",
        "target_label": "Monthly energy use (kWh)",
        "description": "Estimate monthly building energy use from size, occupancy, temperature, insulation, building type, and solar panels.",
        "numeric": ["floor_area_m2", "occupants", "average_temperature_c"],
        "categorical": ["insulation_level", "building_type", "solar_panels"],
        "fields": [
            {"name": "floor_area_m2", "label": "Floor area (m²)", "type": "number", "min": 45, "max": 650, "step": 0.1, "default": 180},
            {"name": "occupants", "label": "Number of occupants", "type": "number", "min": 1, "max": 15, "step": 1, "default": 4},
            {"name": "average_temperature_c", "label": "Average temperature (°C)", "type": "number", "min": 16, "max": 34, "step": 0.1, "default": 27},
            {"name": "insulation_level", "label": "Insulation level", "type": "select", "options": ["Basic", "Standard", "Premium"], "default": "Standard"},
            {"name": "building_type", "label": "Building type", "type": "select", "options": ["Apartment", "Office", "Retail", "House"], "default": "Apartment"},
            {"name": "solar_panels", "label": "Solar panels", "type": "select", "options": ["No", "Yes"], "default": "No"},
        ],
    },
    "delivery": {
        "title": "Last-Mile Delivery Cost",
        "file": ROOT / "datasets" / "urban_delivery_cost.csv",
        "target": "delivery_cost_pkr",
        "target_label": "Estimated delivery cost (PKR)",
        "description": "Estimate urban delivery cost using distance, package weight, traffic, vehicle, priority, and weather.",
        "numeric": ["distance_km", "package_weight_kg"],
        "categorical": ["traffic_level", "vehicle_type", "delivery_priority", "weather"],
        "fields": [
            {"name": "distance_km", "label": "Distance (km)", "type": "number", "min": 0.5, "max": 42, "step": 0.1, "default": 8},
            {"name": "package_weight_kg", "label": "Package weight (kg)", "type": "number", "min": 0.2, "max": 28, "step": 0.1, "default": 3},
            {"name": "traffic_level", "label": "Traffic level", "type": "select", "options": ["Low", "Medium", "High"], "default": "Medium"},
            {"name": "vehicle_type", "label": "Vehicle type", "type": "select", "options": ["Bike", "Van", "Electric Van"], "default": "Van"},
            {"name": "delivery_priority", "label": "Delivery priority", "type": "select", "options": ["Standard", "Express"], "default": "Standard"},
            {"name": "weather", "label": "Weather", "type": "select", "options": ["Clear", "Rain", "Hot"], "default": "Clear"},
        ],
    },
}

ARTIFACTS = ROOT / "artifacts"

def load_dataset(dataset_key):
    """Load one of the bundled CSV datasets."""
    config = DATASETS[dataset_key]
    return pd.read_csv(config["file"])

def build_pipeline(dataset_key):
    """Build a preprocessing + Linear Regression pipeline."""
    config = DATASETS[dataset_key]
    numeric_pipeline = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, config["numeric"]),
        ("categorical", categorical_pipeline, config["categorical"]),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("regression", LinearRegression()),
    ])

def train_one(dataset_key):
    """Train and evaluate one model; return a compact summary."""
    config = DATASETS[dataset_key]
    data = load_dataset(dataset_key)
    features = data.drop(columns=[config["target"]])
    target = data[config["target"]]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=42
    )
    model = build_pipeline(dataset_key)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    summary = {
        "dataset_key": dataset_key,
        "title": config["title"],
        "target": config["target"],
        "rows": int(len(data)),
        "features": list(features.columns),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
        "rmse": round(float(np.sqrt(mse)), 3),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / f"{dataset_key}_linear_regression.joblib")
    # Keep plot generation in training so the web apps can display static charts.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(y_test, predictions, alpha=0.7)
    lower = float(min(y_test.min(), predictions.min()))
    upper = float(max(y_test.max(), predictions.max()))
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted: {config['title']}")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / f"{dataset_key}_actual_vs_predicted.png", dpi=140)
    plt.close(fig)

    residuals = y_test.to_numpy() - predictions
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(predictions, residuals, alpha=0.7)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(f"Residuals: {config['title']}")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / f"{dataset_key}_residuals.png", dpi=140)
    plt.close(fig)
    return summary

def train_all():
    """Train both models and write evaluation results to JSON."""
    summaries = [train_one(key) for key in DATASETS]
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries

def get_model(dataset_key):
    """Load a saved model, training it if the artifacts are missing."""
    model_path = ARTIFACTS / f"{dataset_key}_linear_regression.joblib"
    if not model_path.exists():
        train_all()
    return joblib.load(model_path)

def predict_one(dataset_key, input_values):
    """Make one prediction from a dictionary of raw feature values."""
    config = DATASETS[dataset_key]
    model = get_model(dataset_key)
    row = pd.DataFrame([{name: input_values[name] for name in config["numeric"] + config["categorical"]}])
    return float(model.predict(row)[0])

def dataset_insights(dataset_key):
    """Return small, readable dataset exploration details for the UI."""
    config = DATASETS[dataset_key]
    data = load_dataset(dataset_key)
    return {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "missing": int(data.isna().sum().sum()),
        "target_mean": round(float(data[config["target"]].mean()), 2),
        "target_median": round(float(data[config["target"]].median()), 2),
        "target_min": round(float(data[config["target"]].min()), 2),
        "target_max": round(float(data[config["target"]].max()), 2),
        "describe": data.describe(include="all").fillna("").round(2).to_html(classes="data-table", border=0),
        "preview": data.head(8).round(2).to_html(classes="data-table", index=False, border=0),
        "correlations": data[config["numeric"] + [config["target"]]].corr(numeric_only=True).round(2).to_html(classes="data-table", border=0),
    }
