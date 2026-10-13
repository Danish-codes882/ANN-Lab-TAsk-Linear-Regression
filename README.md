# Urban Analytics Regression Lab

A student-friendly regression project that compares two related urban-operations datasets:
1. **Building energy use**: estimate monthly energy consumption (`monthly_energy_kwh`).
2. **Last-mile delivery cost**: estimate delivery cost in PKR (`delivery_cost_pkr`).

The datasets are **synthetic and generated specifically for this project**. They are designed for practice and are not real customer, business, or building records. Their feature patterns are intentionally realistic enough for learning, but the models must not be used for real-world decisions.

## Features
- Linear Regression with train/test split and preprocessing for numeric and categorical columns.
- Metrics: MAE, RMSE, and R².
- Actual-vs-predicted chart and residual chart.
- Dataset preview, descriptive statistics, and feature correlation.
- Flask web UI using HTML, CSS, and JavaScript.
- Gradio interface for a second UI.
- Reproducible model training and model files saved under `artifacts/`.

## Project structure
```text
Regression_Lab_Project/
├── app.py
├── gradio_app.py
├── train_models.py
├── regression_core.py
├── requirements.txt
├── datasets/
│   ├── urban_building_energy.csv
│   └── urban_delivery_cost.csv
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/app.js
├── reports/
│   └── Regression_Lab_Report.docx
└── artifacts/                 # created after training
```

## 1. Setup on Windows
Open PowerShell inside the project folder:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, run this only for the current terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Train and evaluate both models
```powershell
python train_models.py
```
This saves fitted pipelines and evaluation summaries in `artifacts/`.

## 3. Start the Flask website
```powershell
python app.py
```
Open `http://127.0.0.1:5000` in a browser. Use the form to select a dataset and enter feature values.

## 4. Start the Gradio interface
Open a second terminal in the same project folder, activate the environment, then:
```powershell
python gradio_app.py
```
Open the local URL shown in the terminal, usually `http://127.0.0.1:7860`.

## 5. GitHub upload
Create a repository on GitHub, then run from this folder:
```powershell
git init
git add .
git commit -m "Add urban analytics linear regression lab"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/urban-analytics-regression-lab.git
git push -u origin main
```
Replace the remote URL with your own repository URL. Do not upload `.venv/`, secrets, or unnecessary large files.

## Model notes
Linear Regression is a supervised learning algorithm for predicting a continuous numeric target. The preprocessing pipeline imputes missing numeric values with the median, scales numeric features, imputes categorical values with the most frequent category, and one-hot encodes categories. The data is split into 80% training and 20% testing with a fixed random seed.

Metrics:
- **MAE**: average absolute prediction error, in the target's units.
- **RMSE**: square root of average squared error, in the target's units; larger errors count more.
- **R²**: relative fit score. It can be negative on poor test predictions; 1 is a perfect fit, and 0 is roughly the test-set mean baseline.

## Limitations
- Both CSV datasets are generated synthetic teaching data.
- Linear Regression models a linear relationship after one-hot encoding and may not capture complex nonlinear patterns.
- Evaluation results are specific to this fixed dataset and split.
- Prediction forms are for educational demonstration only.

## Humanization / academic integrity
The code uses clear variable names and comments intended to explain the workflow. Before submitting, run the project yourself, inspect the outputs, understand the metrics, and edit the report to match your instructor's rubric. Do not claim synthetic data is collected from a real city or company.

## Python 3.15 installation note

On some Python 3.15 installations, Gradio's PyYAML/Brotli dependencies may try to compile and fail. Install the core dependencies first using `python -m pip install -r requirements.txt`, then start the Flask interface with `python app.py`. Gradio is intentionally separated from the core requirements for this environment. To try the Gradio interface separately, run `python -m pip install gradio` and then `python gradio_app.py`; if PyYAML/Brotli compilation fails, use Python 3.13 for Gradio or submit the working Flask interface along with the included Gradio source file.
