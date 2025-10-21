from fastapi import FastAPI
from fastapi.responses import JSONResponse,FileResponse
import pickle
import pandas as pd
import numpy as np
import os

app = FastAPI()


MODEL_PATH = "D:/SEMESTER 5/ML Lab/BusinessAcceralation/models/artifacts/2cd_model.pkl"
ARTIFACT_PATH = "D:/SEMESTER 5/ML Lab/BusinessAcceralation/models/artifacts/profit_vs_stress.png"
RESULTS_PATH = "D:/SEMESTER 5/ML Lab/BusinessAcceralation/models/artifacts/2cd_results.pkl"

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
    return {"message": "API 2CD is running! Use /insights, /decisions, /compare, /summary endpoints."}

@app.get("/insights")
def insights():
    if not os.path.exists(RESULTS_PATH):
        return {"error": "Model results not found. Please run model_cd.py first."}
    try:
        with open(RESULTS_PATH, "rb") as f:
            model_output = pickle.load(f)
        insights = model_output.get("insights", [])
        return {"status": "success", "insights": insights}
    except Exception as e:
        return {"error": str(e)}


@app.get("/decisions")
def decisions():
    if not os.path.exists(RESULTS_PATH):
        return {"error": "Results not found. Please run model_cd.py first."}
    try:
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        proj_climate = results.get("proj_climate", pd.DataFrame())
        data = proj_climate[["Project_ID", "Decision", "Stress_Score", "Vulnerability"]]
        return JSONResponse(content=safe_json(data))
    except Exception as e:
        return {"error": str(e)}

@app.get("/compare")
def compare_models():
    if not os.path.exists(RESULTS_PATH):
        return {"error": "Model results not found. Please run model_cd.py first."}
    try:
        with open(RESULTS_PATH, "rb") as f:
            model_output = pickle.load(f)
        compare = model_output.get("compare", pd.DataFrame())
        return JSONResponse(content=safe_json(compare))
    except Exception as e:
        return {"error": str(e)}


@app.get("/summary")
def summary():
    if not os.path.exists(RESULTS_PATH):
        return {"error": "Model results not found. Please run model_cd.py first."}
    try:
        with open(RESULTS_PATH, "rb") as f:
            model_output = pickle.load(f)
        proj_climate = model_output.get("proj_climate", pd.DataFrame())

        decision_counts = proj_climate["Decision"].value_counts().to_dict()
        avg_stress = float(proj_climate["Stress_Score"].mean())
        avg_vuln = float(proj_climate["Vulnerability"].mean())

        return {
            "status": "success",
            "total_projects": len(proj_climate),
            "decision_breakdown": decision_counts,
            "average_stress_score": avg_stress,
            "average_vulnerability": avg_vuln
        }
    except Exception as e:
        return {"error": str(e)}


ARTIFACTS_DIR = "D:/SEMESTER 5/ML Lab/BusinessAcceralation/models/artifacts"

@app.get("/plot/correlation_heatmap")
def plot_correlation_heatmap():
    plot_path = os.path.join(ARTIFACTS_DIR, "2cd_correlation_heatmap.png")
    if not os.path.exists(plot_path):
        return {"error": f"Plot not found at {plot_path}"}
    return FileResponse(plot_path, media_type="image/png")

@app.get("/plot/cost_vs_duration")
def plot_cost_vs_duration():
    plot_path = os.path.join(ARTIFACTS_DIR, "2cd_cost_vs_duration.png")
    if not os.path.exists(plot_path):
        return {"error": f"Plot not found at {plot_path}"}
    return FileResponse(plot_path, media_type="image/png")

@app.get("/plot/pairplots_features")
def plot_pairplots_features():
    plot_path = os.path.join(ARTIFACTS_DIR, "2cd_pairplot_features.png")
    if not os.path.exists(plot_path):
        return {"error": f"Plot not found at {plot_path}"}
    return FileResponse(plot_path, media_type="image/png")
