"""Flask web application for the Urban Analytics Regression Lab."""
from flask import Flask, jsonify, render_template, request
from regression_core import DATASETS, dataset_insights, load_dataset, predict_one, train_all, ARTIFACTS
import json

app = Flask(__name__)

def current_metrics():
    path = ARTIFACTS / "metrics.json"
    if not path.exists():
        train_all()
    return {item["dataset_key"]: item for item in json.loads(path.read_text(encoding="utf-8"))}

@app.route("/")
def home():
    summaries = current_metrics()
    return render_template("index.html", datasets=DATASETS, summaries=summaries)

@app.route("/api/insights/<dataset_key>")
def insights(dataset_key):
    if dataset_key not in DATASETS:
        return jsonify({"error": "Please choose a valid dataset."}), 400
    return jsonify(dataset_insights(dataset_key))

@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    key = payload.get("dataset_key", "energy")
    if key not in DATASETS:
        return jsonify({"error": "Please choose a valid dataset."}), 400
    try:
        values = {}
        for field in DATASETS[key]["fields"]:
            raw = payload.get(field["name"], field["default"])
            if field["type"] == "number":
                values[field["name"]] = float(raw)
                if not (float(field["min"]) <= values[field["name"]] <= float(field["max"])):
                    return jsonify({"error": f"{field['label']} must be between {field['min']} and {field['max']}."}), 400
            else:
                if raw not in field["options"]:
                    return jsonify({"error": f"Choose a valid value for {field['label']}."}), 400
                values[field["name"]] = raw
        prediction = predict_one(key, values)
        return jsonify({
            "prediction": round(prediction, 2),
            "target_label": DATASETS[key]["target_label"],
            "dataset_title": DATASETS[key]["title"],
            "note": "This is an educational estimate from synthetic practice data, not a real-world quote or measurement."
        })
    except (ValueError, TypeError, KeyError) as error:
        return jsonify({"error": f"Please check your inputs. {error}"}), 400

if __name__ == "__main__":
    current_metrics()
    app.run(debug=True)
