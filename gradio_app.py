"""Gradio interface for the same two Linear Regression models."""
import gradio as gr
from regression_core import DATASETS, dataset_insights, load_dataset, predict_one, train_all, ARTIFACTS
import json

def ensure_models():
    if not (ARTIFACTS / "metrics.json").exists():
        train_all()
    return {item["dataset_key"]: item for item in json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))}

METRICS = ensure_models()

def predict_energy(floor_area_m2, occupants, average_temperature_c, insulation_level, building_type, solar_panels):
    values = {
        "floor_area_m2": floor_area_m2, "occupants": occupants,
        "average_temperature_c": average_temperature_c, "insulation_level": insulation_level,
        "building_type": building_type, "solar_panels": solar_panels
    }
    result = predict_one("energy", values)
    return f"Estimated monthly energy use: **{result:,.2f} kWh**\n\nEducational estimate based on synthetic data."

def predict_delivery(distance_km, package_weight_kg, traffic_level, vehicle_type, delivery_priority, weather):
    values = {
        "distance_km": distance_km, "package_weight_kg": package_weight_kg,
        "traffic_level": traffic_level, "vehicle_type": vehicle_type,
        "delivery_priority": delivery_priority, "weather": weather
    }
    result = predict_one("delivery", values)
    return f"Estimated delivery cost: **PKR {result:,.2f}**\n\nEducational estimate based on synthetic data."

def metrics_table():
    rows = [["Dataset", "Rows", "MAE", "RMSE", "R²"]]
    for key, item in METRICS.items():
        rows.append([item["title"], item["rows"], item["mae"], item["rmse"], item["r2"]])
    return rows

with gr.Blocks(title="Urban Analytics Regression Lab", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Urban Analytics Regression Lab\nCompare two Linear Regression models for connected urban-operation problems.")
    with gr.Tab("Building Energy"):
        gr.Markdown("Estimate monthly building energy use from building and environment details.")
        with gr.Row():
            floor_area = gr.Number(label="Floor area (m²)", value=180, minimum=45, maximum=650)
            occupants = gr.Number(label="Occupants", value=4, minimum=1, maximum=15, precision=0)
            temperature = gr.Number(label="Average temperature (°C)", value=27, minimum=16, maximum=34)
        with gr.Row():
            insulation = gr.Dropdown(["Basic", "Standard", "Premium"], value="Standard", label="Insulation")
            building = gr.Dropdown(["Apartment", "Office", "Retail", "House"], value="Apartment", label="Building type")
            solar = gr.Dropdown(["No", "Yes"], value="No", label="Solar panels")
        energy_button = gr.Button("Estimate energy use", variant="primary")
        energy_output = gr.Markdown()
        energy_button.click(predict_energy, [floor_area, occupants, temperature, insulation, building, solar], energy_output)
        gr.Dataframe(load_dataset("energy").head(8), label="Dataset preview", interactive=False)
        gr.Markdown(f"**Test metrics:** MAE {METRICS['energy']['mae']} kWh | RMSE {METRICS['energy']['rmse']} kWh | R² {METRICS['energy']['r2']}")
    with gr.Tab("Delivery Cost"):
        gr.Markdown("Estimate delivery cost using distance, weight, traffic, vehicle, priority, and weather.")
        with gr.Row():
            distance = gr.Number(label="Distance (km)", value=8, minimum=0.5, maximum=42)
            weight = gr.Number(label="Package weight (kg)", value=3, minimum=0.2, maximum=28)
        with gr.Row():
            traffic = gr.Dropdown(["Low", "Medium", "High"], value="Medium", label="Traffic")
            vehicle = gr.Dropdown(["Bike", "Van", "Electric Van"], value="Van", label="Vehicle")
            priority = gr.Dropdown(["Standard", "Express"], value="Standard", label="Priority")
            weather = gr.Dropdown(["Clear", "Rain", "Hot"], value="Clear", label="Weather")
        delivery_button = gr.Button("Estimate delivery cost", variant="primary")
        delivery_output = gr.Markdown()
        delivery_button.click(predict_delivery, [distance, weight, traffic, vehicle, priority, weather], delivery_output)
        gr.Dataframe(load_dataset("delivery").head(8), label="Dataset preview", interactive=False)
        gr.Markdown(f"**Test metrics:** MAE PKR {METRICS['delivery']['mae']} | RMSE PKR {METRICS['delivery']['rmse']} | R² {METRICS['delivery']['r2']}")
    with gr.Tab("Model Comparison"):
        gr.Markdown("Metrics are calculated on a held-out 20% test split. Lower MAE/RMSE is better; R² closer to 1 is generally better.")
        gr.Dataframe(metrics_table(), headers=["Dataset", "Rows", "MAE", "RMSE", "R²"], interactive=False)
        gr.Markdown("Both datasets are synthetic teaching data. These results demonstrate the workflow and are not real-world benchmarks.")

if __name__ == "__main__":
    demo.launch()
