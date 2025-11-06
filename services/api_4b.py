from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import pandas as pd
import json
import os
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any, List

app = FastAPI()

RESULTS_FILE = r"C:\new\BusinessExpansionAndAcceleration\models\artifacts\4b_results.pkl"
ARTIFACTS_DIR = os.path.dirname(RESULTS_FILE)

try:
    with open(RESULTS_FILE, "rb") as f:
        results_4b = pickle.load(f)
    print(f"Successfully loaded 4b results from {RESULTS_FILE}")
except FileNotFoundError:
    print(f"ERROR: Could not load 4b results file from {RESULTS_FILE}")
    results_4b = {
        "metrics": {"scratch": {}, "inbuilt": {}},
        "best_model_name": "Unknown",
        "plot_paths": {},
        "processed_data_top5": []
    }
except Exception as e:
    print(f"An unexpected error occurred loading {RESULTS_FILE}: {e}")
    results_4b = {}

def safe_json(df: pd.DataFrame):
    df = df.copy()
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    df = df.where(pd.notnull(df), None)
    
    def convert_value(x):
        if x is None:
            return None
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (np.floating, float)):
            if np.isfinite(x):
                return float(x)
            else:
                return None
        return str(x)
    
    return df.map(convert_value).to_dict(orient="records")

@app.get("/")
def root():
    return {
        "message": "API for 4B (Asset ROI Prediction) is running!",
        "endpoints": [
            "/metrics",
            "/best_model",
            "/processed_data_summary",
            "/plot/roi_by_performance",
            "/plot/utilization_vs_roi",
            "/plot/actual_vs_predicted"
        ]
    }

@app.get("/metrics")
def get_metrics():
    return results_4b.get("metrics", {"error": "Metrics not found"})

@app.get("/best_model")
def get_best_model():
    return {"best_model_name": results_4b.get("best_model_name", "Unknown")}

@app.get("/processed_data_summary")
def get_processed_data_summary():
    try:
        data = results_4b.get("processed_data_top5", [])
        if not data:
            return []
        df = pd.DataFrame(data)
        return JSONResponse(content=safe_json(df))
    except Exception as e:
        return {"error": str(e)}

def get_plot_path(plot_name: str):
    plot_paths = results_4b.get("plot_paths", {})
    
    if plot_name in plot_paths:
        plot_path = plot_paths[plot_name]
        if os.path.exists(plot_path):
            return plot_path
    
    candidate_path = os.path.join(ARTIFACTS_DIR, f"{plot_name}.png")
    if os.path.exists(candidate_path):
        return candidate_path
        
    candidate_path_4b = os.path.join(ARTIFACTS_DIR, f"4b_{plot_name}.png")
    if os.path.exists(candidate_path_4b):
        return candidate_path_4b

    raise HTTPException(
        status_code=404,
        detail={
            "error": f"Plot '{plot_name}' not found.",
            "available_plots": list(plot_paths.keys())
        }
    )

@app.get("/plot/{plot_name}")
def get_plot(plot_name: str):
    try:
        plot_path = get_plot_path(plot_name)
        return FileResponse(
            plot_path,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={os.path.basename(plot_path)}"}
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))