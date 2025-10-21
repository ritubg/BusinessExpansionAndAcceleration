from fastapi import FastAPI
from fastapi.responses import JSONResponse,FileResponse
import pickle
import os
import pandas as pd
import numpy as np

app = FastAPI()


MODEL_PATH = "D:/SEMESTER 5/ML Lab/BusinessAcceralation/models/artifacts/3b_model.pkl"
RESULTS_PATH = "D:/SEMESTER 5/ML Lab/BusinessAcceralation/models/artifacts/3b_results.pkl"
ARTIFACTS_DIR = "D:/SEMESTER 5/ML Lab/BusinessAcceralation/models/artifacts"

PLOT_FILES = {
    "avg_profitability_per_cluster": os.path.join(ARTIFACTS_DIR, "3b_avg_profitability_per_cluster.png"),
    "cluster_count": os.path.join(ARTIFACTS_DIR, "3b_cluster_count.png"),
    "cluster_scatter": os.path.join(ARTIFACTS_DIR, "3b_cluster_scatter.png"),
    "feature_heatmap": os.path.join(ARTIFACTS_DIR, "3b_feature_heatmap.png"),
    "profitability_boxplot": os.path.join(ARTIFACTS_DIR, "3b_profitability_boxplot.png")
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
    return {"message": "API 3B (Client Segmentation) is running! Use /clusters, /summary, /best_k endpoints."}

@app.get("/clusters")
def clusters():
    if not os.path.exists(RESULTS_PATH):
        return {"error": "Results not found. Please run model_3b.py first."}
    try:
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        data = results.get("data", pd.DataFrame())
        cols = ["Project_ID", "Client_ID", "Cluster", "Profitability", "Contract_Value_INR", "Project_Size_sq_ft"]
        return JSONResponse(content=safe_json(data[cols]))
    except Exception as e:
        return {"error": str(e)}


@app.get("/summary")
def summary():
    if not os.path.exists(RESULTS_PATH):
        return {"error": "Results not found. Please run model_3b.py first."}
    try:
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        cluster_summary = results.get("cluster_summary", pd.DataFrame())
        return JSONResponse(content=safe_json(cluster_summary))
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
