from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import pandas as pd
import json
import os
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any, List

app = FastAPI()

RESULTS_FILE = r"C:\new\BusinessExpansionAndAcceleration\models\artifacts\1f_results.pkl"
ARTIFACTS_DIR = os.path.dirname(RESULTS_FILE)

try:
    with open(RESULTS_FILE, "rb") as f:
        results_1f = pickle.load(f)
    print(f"Successfully loaded 1f results from {RESULTS_FILE}")
except FileNotFoundError:
    print(f"ERROR: Could not load 1f results file from {RESULTS_FILE}")
    results_1f = {
        "metrics": {"classification": {}, "clustering": {}, "anomaly": {}},
        "plot_paths": {},
        "risk_summary_top5": [],
        "cluster_summary_top5": []
    }
except Exception as e:
    print(f"An unexpected error occurred loading {RESULTS_FILE}: {e}")
    results_1f = {}

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
        "message": "API for 1F (Financial Analysis) is running!",
        "endpoints": [
            "/metrics",
            "/risk_summary",
            "/cluster_summary",
            "/plot/risk_distribution",
            "/plot/risk_scatter",
            "/plot/cluster_distribution",
            "/plot/cluster_scatter",
            "/plot/anomaly_distribution"
        ]
    }

@app.get("/metrics")
def get_metrics():
    return results_1f.get("metrics", {"error": "Metrics not found"})

@app.get("/risk_summary")
def get_risk_summary():
    try:
        data = results_1f.get("risk_summary_top5", [])
        if not data:
            return []
        df = pd.DataFrame(data)
        return JSONResponse(content=safe_json(df))
    except Exception as e:
        return {"error": str(e)}

@app.get("/cluster_summary")
def get_cluster_summary():
    try:
        data = results_1f.get("cluster_summary_top5", [])
        if not data:
            return []
        df = pd.DataFrame(data)
        return JSONResponse(content=safe_json(df))
    except Exception as e:
        return {"error": str(e)}

def get_plot_path(plot_name: str):
    plot_paths = results_1f.get("plot_paths", {})
    
    if plot_name in plot_paths:
        plot_path = plot_paths[plot_name]
        if os.path.exists(plot_path):
            return plot_path
    
    candidate_path = os.path.join(ARTIFACTS_DIR, f"{plot_name}.png")
    if os.path.exists(candidate_path):
        return candidate_path
        
    candidate_path_1f = os.path.join(ARTIFACTS_DIR, f"1f_{plot_name}.png")
    if os.path.exists(candidate_path_1f):
        return candidate_path_1f

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