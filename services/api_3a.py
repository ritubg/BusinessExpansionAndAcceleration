from fastapi import FastAPI
from fastapi.responses import JSONResponse,FileResponse
import pandas as pd
import numpy as np
import pickle
import os

app = FastAPI()

MODEL_PATH = "D:/SEMESTER 5/ML Lab/BusinessAcceralation/models/artifacts/3a_model.pkl"
RESULTS_PATH = "D:/SEMESTER 5/ML Lab/BusinessAcceralation/models/artifacts/3a_results.pkl"
ARTIFACTS_DIR = "D:/SEMESTER 5/ML Lab/BusinessAcceralation/models/artifacts"
PLOT_FILES = {
    "churn_distribution": os.path.join(ARTIFACTS_DIR, "3a_churn_distribution.png"),
    "value_vs_churn": os.path.join(ARTIFACTS_DIR, "3a_value_vs_churn.png"),
    "action_summary": os.path.join(ARTIFACTS_DIR, "3a_action_summary.png"),
    "feature_heatmap": os.path.join(ARTIFACTS_DIR, "3a_feature_heatmap.png")
}

def safe_json(df: pd.DataFrame):
    """Convert DataFrame to JSON-safe list of dicts."""
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    def convert_value(x):
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (np.floating, float)):
            return float(x)
        if isinstance(x, str):
            return x
        return str(x)
    
    return df.applymap(convert_value).to_dict(orient="records")


@app.get("/")
def root():
    return {"message": "API 3A is running! Use /metrics, /report, /top_churn endpoints."}


@app.get("/metrics")
def metrics():
    if not os.path.exists(RESULTS_PATH):
        return {"error": "Results not found. Please run 3a_model.py first."}
    try:
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        return JSONResponse(content={
            "status": "success",
            "metrics": results.get("metrics", {}),
            "best_model_name": results.get("best_model_name", "")
        })
    except Exception as e:
        return {"error": str(e)}


@app.get("/report")
def report():
    if not os.path.exists(RESULTS_PATH):
        return {"error": "Results not found. Please run 3a_model.py first."}
    try:
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        client_report = results.get("client_report", pd.DataFrame())
        return JSONResponse(content=safe_json(client_report))
    except Exception as e:
        return {"error": str(e)}


@app.get("/top_churn")
def top_churn(n: int = 10):
    """Return top N clients with highest churn probability"""
    if not os.path.exists(RESULTS_PATH):
        return {"error": "Results not found. Please run 3a_model.py first."}
    try:
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        client_report = results.get("client_report", pd.DataFrame())
        top_clients = client_report.sort_values("Churn_Prob", ascending=False).head(n)
        return JSONResponse(content=safe_json(top_clients))
    except Exception as e:
        return {"error": str(e)}


@app.get("/plots/{plot_name}")
def get_plot(plot_name: str):
    """Return individual plot by name (e.g., /plots/model_accuracy)."""
    if plot_name not in PLOT_FILES:
        return {"error": f"Plot '{plot_name}' not found. Available: {list(PLOT_FILES.keys())}"}
    file_path = PLOT_FILES[plot_name]
    if not os.path.exists(file_path):
        return {"error": f"Plot file not found: {file_path}"}
    return FileResponse(file_path, media_type="image/png")
