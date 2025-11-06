from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import pandas as pd
import json
import os
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any, List

app = FastAPI()

RESULTS_FILE = r"C:\new\BusinessExpansionAndAcceleration\models\artifacts\1c_results.pkl"
ARTIFACTS_DIR = os.path.dirname(RESULTS_FILE)

try:
    with open(RESULTS_FILE, "rb") as f:
        results_1c = pickle.load(f)
    print(f"Successfully loaded 1c results from {RESULTS_FILE}")
except FileNotFoundError:
    print(f"ERROR: Could not load 1c results file from {RESULTS_FILE}")
    results_1c = {
        "metrics": {"scratch": {}, "inbuilt": {}},
        "best_model_name": "Unknown",
        "plot_paths": {},
        "zone_project_distribution": {},
        "forecast_summary": []
    }
except Exception as e:
    print(f"An unexpected error occurred loading {RESULTS_FILE}: {e}")
    results_1c = {}

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
        "message": "API for 1C (Cost Forecasting) is running!",
        "endpoints": [
            "/metrics",
            "/best_model",
            "/forecast_summary",
            "/zone_distribution",
            "/plot/cost_forecast_map",
            "/plot/project_distribution_pies"
        ]
    }

@app.get("/metrics")
def get_metrics():
    return results_1c.get("metrics", {"error": "Metrics not found"})

@app.get("/best_model")
def get_best_model():
    return {"best_model_name": results_1c.get("best_model_name", "Unknown")}

@app.get("/forecast_summary")
def get_forecast_summary():
    try:
        data = results_1c.get("forecast_summary", [])
        if not data:
            return []
        df = pd.DataFrame(data)
        return JSONResponse(content=safe_json(df))
    except Exception as e:
        return {"error": str(e)}

@app.get("/zone_distribution")
def get_zone_distribution():
    return results_1c.get("zone_project_distribution", {"error": "Zone distribution data not found"})

def get_plot_path(plot_name: str):
    plot_paths = results_1c.get("plot_paths", {})
    
    if plot_name in plot_paths:
        plot_path = plot_paths[plot_name]
        if os.path.exists(plot_path):
            return plot_path
    
    candidate_path = os.path.join(ARTIFACTS_DIR, f"{plot_name}.png")
    if os.path.exists(candidate_path):
        return candidate_path
        
    candidate_path_1c = os.path.join(ARTIFACTS_DIR, f"1c_{plot_name}.png")
    if os.path.exists(candidate_path_1c):
        return candidate_path_1c

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