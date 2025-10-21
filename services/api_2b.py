from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd
import json
from fastapi.responses import JSONResponse
from fastapi import HTTPException
import os
from fastapi.responses import FileResponse

app = FastAPI()

with open("D:/SEMESTER 5/ML Lab/BusinessAcceralation/models/artifacts/2b_results.pkl", "rb") as f:
    results = pickle.load(f)
def safe_json(df: pd.DataFrame):
    """Convert DataFrame to JSON-safe list of dicts"""
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
    
    return df.applymap(convert_value).to_dict(orient="records")

@app.get("/")
def root():
    return {"message": "API is running! Use /metrics, /top_risky_cost or /safe_projects_cost endpoints."}

@app.get("/metrics")
def get_metrics():
    return {
        "cost_rmse": float(results["cost_rmse"]) if np.isfinite(results["cost_rmse"]) else None,
        "timeline_rmse": float(results["timeline_rmse"]) if np.isfinite(results["timeline_rmse"]) else None
    }


from fastapi.responses import JSONResponse

@app.get("/top_risky_cost")
def top_risky_cost():
    try:
        return JSONResponse(content=safe_json(results["top_risky_cost"]))
    except Exception as e:
        return {"error": str(e)}

@app.get("/safe_projects_cost")
def safe_projects_cost():
    try:
        return JSONResponse(content=safe_json(results["safe_projects_cost"]))
    except Exception as e:
        return {"error": str(e)}

ARTIFACTS_DIR = "D:/SEMESTER 5/ML Lab/BusinessAcceralation/models/artifacts"

def get_plot_path(plot_name: str):
    plot_paths = results.get("plot_paths", {})
    if plot_name in plot_paths:
        plot_path = plot_paths[plot_name]
        if os.path.exists(plot_path):
            return plot_path
    
    candidate_path = os.path.join(ARTIFACTS_DIR, f"{plot_name}.png")
    if os.path.exists(candidate_path):
        return candidate_path

    raise HTTPException(
        status_code=404,
        detail={
            "error": f"Plot '{plot_name}' not found.",
            "available_plots": list(plot_paths.keys())
        }
    )

@app.get("/plot/{plot_name}")
def get_plot(plot_name: str):
    """
    Retrieve a specific plot by name.
    Example: /plot/2b_cost_predictions
    """
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
