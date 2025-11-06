from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import pandas as pd
import json
import os
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any, List

app = FastAPI()

RESULTS_FILE = r"C:\new\BusinessExpansionAndAcceleration\models\artifacts\1d_results.pkl"
ARTIFACTS_DIR = os.path.dirname(RESULTS_FILE)

try:
    with open(RESULTS_FILE, "rb") as f:
        results_1d = pickle.load(f)
    print(f"Successfully loaded 1d results from {RESULTS_FILE}")
except FileNotFoundError:
    print(f"ERROR: Could not load 1d results file from {RESULTS_FILE}")
    results_1d = {
        "metrics": {"clustering": {}, "classification": {}},
        "plot_paths": {},
        "competitor_profiles_top5": [],
        "competitor_similarity_top10": [],
        "market_share_top10": [],
        "association_rules_top5": [],
        "seasonality_top5": []
    }
except Exception as e:
    print(f"An unexpected error occurred loading {RESULTS_FILE}: {e}")
    results_1d = {}

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

def safe_json_list_of_dicts(data: List[Dict]) -> List[Dict]:
    try:
        df = pd.DataFrame(data)
        return safe_json(df)
    except Exception:
        return data

@app.get("/")
def root():
    return {
        "message": "API for 1D (Competitor Analysis) is running!",
        "endpoints": [
            "/metrics",
            "/competitor_profiles",
            "/competitor_similarity",
            "/market_share",
            "/association_rules",
            "/seasonality",
            "/plot/clustering_plot",
            "/plot/market_share_plot",
            "/plot/confusion_matrix"
        ]
    }

@app.get("/metrics")
def get_metrics():
    return results_1d.get("metrics", {"error": "Metrics not found"})

@app.get("/competitor_profiles")
def get_competitor_profiles():
    try:
        data = results_1d.get("competitor_profiles_top5", [])
        return JSONResponse(content=safe_json_list_of_dicts(data))
    except Exception as e:
        return {"error": str(e)}

@app.get("/competitor_similarity")
def get_competitor_similarity():
    try:
        data = results_1d.get("competitor_similarity_top10", [])
        return JSONResponse(content=safe_json_list_of_dicts(data))
    except Exception as e:
        return {"error": str(e)}

@app.get("/market_share")
def get_market_share():
    try:
        data = results_1d.get("market_share_top10", [])
        return JSONResponse(content=safe_json_list_of_dicts(data))
    except Exception as e:
        return {"error": str(e)}

@app.get("/association_rules")
def get_association_rules():
    try:
        data = results_1d.get("association_rules_top5", [])
        return JSONResponse(content=safe_json_list_of_dicts(data))
    except Exception as e:
        return {"error": str(e)}

@app.get("/seasonality")
def get_seasonality():
    try:
        data = results_1d.get("seasonality_top5", [])
        return JSONResponse(content=safe_json_list_of_dicts(data))
    except Exception as e:
        return {"error": str(e)}

def get_plot_path(plot_name: str):
    plot_paths = results_1d.get("plot_paths", {})
    
    if plot_name in plot_paths:
        plot_path = plot_paths[plot_name]
        if os.path.exists(plot_path):
            return plot_path
    
    candidate_path = os.path.join(ARTIFACTS_DIR, f"{plot_name}.png")
    if os.path.exists(candidate_path):
        return candidate_path
        
    candidate_path_1d = os.path.join(ARTIFACTS_DIR, f"1d_{plot_name}.png")
    if os.path.exists(candidate_path_1d):
        return candidate_path_1d

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