from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.model_2a import Project2AModel

app = FastAPI(
    title="Project Risk Analysis API",
    description="API for predicting project cost overruns and delays",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "C:/new/BusinessExpansionAndAcceleration/models/artifacts/2a_model.pkl"
RESULTS_PATH = "C:/new/BusinessExpansionAndAcceleration/models/artifacts/2a_results.pkl"
print(os.path.exists(RESULTS_PATH))
DATA_PATH = "C:/new/BusinessExpansionAndAcceleration/models/projects_history.csv"
print(os.path.exists(DATA_PATH))
ARTIFACTS_DIR = "C:/new//BusinessExpansionAndAcceleration/models/artifacts"

def safe_json(df: pd.DataFrame):
    """Convert DataFrame to JSON-safe list of dicts"""
    df = df.copy()
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    df = df.fillna(0)
    
    def convert_value(x):
        if x is None:
            return None
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (np.floating, float)):
            return float(x)
        if isinstance(x, str):
            return x
        return str(x)
    
    return df.applymap(convert_value).to_dict(orient="records")


@app.get("/", tags=["Root"])
def root():
    """Root endpoint with API documentation"""
    return {
        "message": "API 2A - Project Risk Analysis System",
        "version": "1.0.0",
        "endpoints": {
            "training": {
                "/train": "Train the prediction models",
                "/predict": "Run predictions and generate plots"
            },
            "data_endpoints": {
                "/metrics": "Get RMSE and average overrun metrics",
                "/top_risky": "Get top 5 projects with highest cost overrun",
                "/safe_projects": "Get projects with no predicted overrun",
                "/top_delay": "Get top 5 projects with highest delay probability",
                "/at_risk": "Get top 10 projects at risk (combined score)",
                "/insights": "Get analytical insights and statistics"
            },
            "visualization": {
                "/plots": "List all available plot paths",
                "/plot/{plot_name}": "Get specific plot image",
                "/plot_list": "Get list of all plot names"
            },
            "reports": {
                "/summary": "Get complete summary of all results",
                "/full_report": "Get comprehensive analysis report"
            }
        },
        "usage": "Start with /train, then /predict, then access data endpoints"
    }


@app.get("/train", tags=["Training"])
def train():
    """Train cost prediction and delay prediction models"""
    try:
        if not os.path.exists(DATA_PATH):
            raise HTTPException(status_code=404, detail=f"Data file not found at {DATA_PATH}")
        
        model = Project2AModel(csv_path=DATA_PATH)
        model.train_models(model_path=MODEL_PATH)
        
        return {
            "status": "success",
            "message": "Models trained successfully",
            "model_path": MODEL_PATH,
            "models": ["Cost Prediction (Linear Regression)", "Delay Prediction (Logistic Regression)"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict", tags=["Training"])
def predict():
    """Run predictions on all projects and generate visualizations"""
    try:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=404, detail="Model not found. Please run /train first")
        
        if not os.path.exists(DATA_PATH):
            raise HTTPException(status_code=404, detail=f"Data file not found at {DATA_PATH}")
        
        model = Project2AModel(csv_path=DATA_PATH)
        results = model.run_predictions(model_path=MODEL_PATH, savepath=RESULTS_PATH)
        
        num_plots = len(results.get("plot_paths", {}))
        
        return {
            "status": "success",
            "message": "Predictions completed successfully",
            "results_path": RESULTS_PATH,
            "plots_generated": num_plots,
            "plot_names": list(results.get("plot_paths", {}).keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Data"])
def metrics():
    """Get model performance metrics"""
    try:
        if not os.path.exists(RESULTS_PATH):
            raise HTTPException(status_code=404, detail="Results not found. Run /train and /predict first")
        
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        
        rmse = results.get("rmse", None)
        overall_avg = results.get("overall_avg", None)
        
        rmse = float(rmse) if rmse is not None and np.isfinite(rmse) else None
        overall_avg = float(overall_avg) if overall_avg is not None and np.isfinite(overall_avg) else None
        
        return {
            "rmse_cost": rmse,
            "overall_avg_overrun_percent": overall_avg,
            "description": {
                "rmse_cost": "Root Mean Square Error for cost predictions",
                "overall_avg_overrun_percent": "Average predicted cost overrun percentage"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/top_risky", tags=["Data"])
def top_risky():
    """Get top 5 projects with highest predicted cost overrun"""
    try:
        if not os.path.exists(RESULTS_PATH):
            raise HTTPException(status_code=404, detail="Results not found. Run /train and /predict first")
        
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        
        return {
            "data": safe_json(results["top_risky"]),
            "count": len(results["top_risky"]),
            "description": "Top 5 projects with highest predicted cost overrun percentage"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/safe_projects", tags=["Data"])
def safe_projects():
    """Get projects with no predicted cost overrun"""
    try:
        if not os.path.exists(RESULTS_PATH):
            raise HTTPException(status_code=404, detail="Results not found. Run /train and /predict first")
        
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        
        return {
            "data": safe_json(results["safe_projects"]),
            "count": len(results["safe_projects"]),
            "description": "Projects with predicted cost overrun <= 0%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/top_delay", tags=["Data"])
def top_delay():
    """Get top 5 projects with highest delay probability"""
    try:
        if not os.path.exists(RESULTS_PATH):
            raise HTTPException(status_code=404, detail="Results not found. Run /train and /predict first")
        
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        
        return {
            "data": safe_json(results["top_delay"]),
            "count": len(results["top_delay"]),
            "description": "Top 5 projects with highest probability of delay"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/at_risk", tags=["Data"])
def at_risk():
    """Get top 10 projects at highest combined risk"""
    try:
        if not os.path.exists(RESULTS_PATH):
            raise HTTPException(status_code=404, detail="Results not found. Run /train and /predict first")
        
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        
        return {
            "data": safe_json(results["at_risk"]),
            "count": len(results["at_risk"]),
            "description": "Top 10 projects with highest combined risk score (cost overrun + delay)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/insights", tags=["Data"])
def insights():
    """Get additional analytical insights and statistics"""
    try:
        if not os.path.exists(RESULTS_PATH):
            raise HTTPException(status_code=404, detail="Results not found. Run /train and /predict first")
        
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        
        insights_data = results.get("insights", {})
        
        return {
            "insights": insights_data,
            "description": "Statistical insights about project risks"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plot/{plot_name}", tags=["Visualization"])
def get_plot(plot_name: str):
    """Return PNG plot by name"""
    if not os.path.exists(RESULTS_PATH):
        raise HTTPException(status_code=404, detail="Results not found. Run 2A first.")

    with open(RESULTS_PATH, "rb") as f:
        results = pickle.load(f)

    plot_paths = results.get("plot_paths", {})
    if plot_name not in plot_paths:
        raise HTTPException(status_code=404, detail={
            "error": f"Plot '{plot_name}' not found",
            "available_plots": list(plot_paths.keys())
        })

    plot_file = plot_paths[plot_name]
    if not os.path.exists(plot_file):
        raise HTTPException(status_code=404, detail=f"Plot file missing at {plot_file}")

    return FileResponse(plot_file, media_type="image/png", headers={"Content-Disposition": f"inline; filename={plot_name}.png"})
    

@app.get("/plot_list", tags=["Visualization"])
def plot_list():
    """Get simple list of all available plot names"""
    try:
        if not os.path.exists(RESULTS_PATH):
            raise HTTPException(status_code=404, detail="Results not found. Run /train and /predict first")
        
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        
        plot_paths = results.get("plot_paths", {})
        
        return {
            "available_plots": list(plot_paths.keys()),
            "count": len(plot_paths),
            "usage": "Access plots via /plot/{plot_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/plot/{plot_name}", tags=["Visualization"])
def get_plot(plot_name: str):
    """Retrieve a specific plot image
    
    Available plots:
    - predicted_vs_actual
    - overrun_distribution
    - delay_distribution
    - risk_heatmap
    - combined_risk_distribution
    - duration_scatter
    - top_risky_bar
    - residual_plot
    """
    try:
        if not os.path.exists(RESULTS_PATH):
            raise HTTPException(status_code=404, detail="Results not found. Run /train and /predict first")
        
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        
        plot_paths = results.get("plot_paths", {})
        
        if plot_name not in plot_paths:
            available = list(plot_paths.keys())
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Plot '{plot_name}' not found",
                    "available_plots": available
                }
            )
        
        plot_path = plot_paths[plot_name]
        
        if not os.path.exists(plot_path):
            raise HTTPException(
                status_code=404,
                detail=f"Plot file not found at {plot_path}"
            )
        
        return FileResponse(
            plot_path,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={plot_name}.png"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary", tags=["Reports"])
def summary():
    """Get a complete summary of all results"""
    try:
        if not os.path.exists(RESULTS_PATH):
            raise HTTPException(status_code=404, detail="Results not found. Run /train and /predict first")
        
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        
        return {
            "metrics": {
                "rmse": float(results.get("rmse", 0)),
                "overall_avg_overrun": float(results.get("overall_avg", 0))
            },
            "insights": results.get("insights", {}),
            "summary": {
                "top_risky_count": len(results.get("top_risky", [])),
                "safe_projects_count": len(results.get("safe_projects", [])),
                "top_delay_count": len(results.get("top_delay", [])),
                "at_risk_count": len(results.get("at_risk", []))
            },
            "plots_available": list(results.get("plot_paths", {}).keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/full_report", tags=["Reports"])
def full_report():
    """Get comprehensive analysis report with all data"""
    try:
        if not os.path.exists(RESULTS_PATH):
            raise HTTPException(status_code=404, detail="Results not found. Run /train and /predict first")
        
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        
        return {
            "status": "success",
            "metrics": {
                "rmse": float(results.get("rmse", 0)),
                "overall_avg_overrun_percent": float(results.get("overall_avg", 0))
            },
            "insights": results.get("insights", {}),
            "top_risky_projects": safe_json(results.get("top_risky", pd.DataFrame())),
            "safe_projects": safe_json(results.get("safe_projects", pd.DataFrame())),
            "top_delay_projects": safe_json(results.get("top_delay", pd.DataFrame())),
            "at_risk_projects": safe_json(results.get("at_risk", pd.DataFrame())),
            "plot_paths": results.get("plot_paths", {}),
            "recommendations": generate_recommendations(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", tags=["System"])
def health_check():
    """Check API health and data availability"""
    return {
        "status": "healthy",
        "files": {
            "model_exists": os.path.exists(MODEL_PATH),
            "results_exist": os.path.exists(RESULTS_PATH),
            "data_exists": os.path.exists(DATA_PATH)
        },
        "paths": {
            "model": MODEL_PATH,
            "results": RESULTS_PATH,
            "data": DATA_PATH,
            "artifacts": ARTIFACTS_DIR
        }
    }


def get_plot_description(plot_name: str) -> str:
    """Get description for each plot type"""
    descriptions = {
        "predicted_vs_actual": "Scatter plot comparing predicted vs actual costs",
        "overrun_distribution": "Histogram showing distribution of cost overruns",
        "delay_distribution": "Histogram showing distribution of delay probabilities",
        "risk_heatmap": "Heatmap of top 20 high-risk projects",
        "combined_risk_distribution": "Histogram showing combined risk score distribution",
        "duration_scatter": "Scatter plot of planned vs actual project duration",
        "top_risky_bar": "Bar chart of top 10 high-risk projects",
        "residual_plot": "Residual plot showing prediction errors"
    }
    return descriptions.get(plot_name, "Visualization plot")


def generate_recommendations(results: dict) -> dict:
    """Generate actionable recommendations based on results"""
    insights = results.get("insights", {})
    
    recommendations = []
    
    high_risk = insights.get("high_risk_projects", 0)
    if high_risk > 0:
        recommendations.append({
            "priority": "HIGH",
            "category": "Risk Management",
            "message": f"{high_risk} projects identified as high-risk. Immediate review required."
        })
    
    avg_delay = insights.get("avg_delay_probability", 0)
    if avg_delay > 0.5:
        recommendations.append({
            "priority": "MEDIUM",
            "category": "Schedule Management",
            "message": f"Average delay probability is {avg_delay:.2%}. Consider timeline reviews."
        })
    
    on_budget = insights.get("projects_on_budget", 0)
    total = insights.get("total_projects", 1)
    if on_budget / total < 0.3:
        recommendations.append({
            "priority": "HIGH",
            "category": "Budget Control",
            "message": f"Only {on_budget}/{total} projects on budget. Budget controls need strengthening."
        })
    
    return {
        "recommendations": recommendations,
        "total_recommendations": len(recommendations)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)