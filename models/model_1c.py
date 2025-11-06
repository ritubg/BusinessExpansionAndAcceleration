import os
import warnings
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any, Tuple, List

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

class CostForecastingPipeline:
    
    def __init__(self, raw_dir: str, zone_dir: str):
        print(f"[1C] Initializing pipeline and loading data...")
        self.raw_dir = raw_dir
        self.zone_dir = zone_dir
        
        try:
            self.projects_df, self.market_df, self.zones_gdf = self._load_data(raw_dir, zone_dir)
            print(f"[1C] Loaded {len(self.projects_df)} project records and {len(self.market_df)} market records.")
        except FileNotFoundError as e:
            print(f"[1C] ERROR: Could not load input files.")
            print(f"[1C] Full error: {e}")
            raise

        self.processed_df = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def _load_data(self, raw_dir: str, zone_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
        projects_df = pd.read_csv(os.path.join(raw_dir, "projects_history.csv"))
        market_df = pd.read_csv(os.path.join(raw_dir, "market_demand.csv"))
        zones_gdf = gpd.read_file(zone_dir)
        return projects_df, market_df, zones_gdf

    def _preprocess_data(self, projects_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        market_df_copy = market_df.copy()
        projects_df_copy = projects_df.copy()
        
        market_df_copy.columns = market_df_copy.columns.str.strip().str.replace(" ", "_")
        market_df_copy["Ward_id"] = market_df_copy["Ward_id"].astype(int)
        projects_df_copy["Location_Ward"] = projects_df_copy["Location_Ward"].astype(int)
        
        merged_df = projects_df_copy.merge(
            market_df_copy[["Ward_id", "Zone_Name", "Demand_Index"]],
            left_on="Location_Ward",
            right_on="Ward_id",
            how="left"
        )
        
        merged_df["Contractor_ID"] = merged_df["Contractor_ID"].fillna("UNKNOWN")
        merged_df["Demand_Index"] = merged_df["Demand_Index"].fillna(merged_df["Demand_Index"].median())
        merged_df.dropna(subset=["Planned_Cost_INR", "Zone_Name"], inplace=True)
        
        return merged_df

    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"mse": float(mse), "r2": float(r2)}

    def _train_scratch_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        class ScratchLinearRegression:
            def fit(self, X, y):
                X_b = np.c_[np.ones((X.shape[0], 1)), X]
                theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
                self.intercept_, self.coef_ = theta_best[0], theta_best[1:]
            def predict(self, X):
                return X.dot(self.coef_) + self.intercept_

        slr = ScratchLinearRegression()
        slr.fit(X_train.values, y_train.values)
        y_pred_lr = slr.predict(X_test.values)
        lr_metrics = self._evaluate_model(y_test, y_pred_lr)
        
        return {"LinearRegression": lr_metrics, "RandomForest": {"mse": 0.0, "r2": 0.0}}

    def _train_inbuilt_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        categorical_features = ["Project_Type"]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )

        lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
        lr_pipeline.fit(X_train, y_train)
        y_pred_lr = lr_pipeline.predict(X_test)
        metrics_lr = self._evaluate_model(y_test, y_pred_lr)

        rf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
        ])
        rf_pipeline.fit(X_train, y_train)
        y_pred_rf = rf_pipeline.predict(X_test)
        metrics_rf = self._evaluate_model(y_test, y_pred_rf)
        
        self.models = {
            "LinearRegression": lr_pipeline,
            "RandomForest": rf_pipeline
        }
        
        return {"LinearRegression": metrics_lr, "RandomForest": metrics_rf}

    def _generate_plots(self, artifacts_dir: str) -> Dict[str, Any]:
        os.makedirs(artifacts_dir, exist_ok=True)
        plot_paths = {}
        sns.set_style("whitegrid")

        zone_forecast = self.processed_df.groupby("Zone_Name").agg(Forecasted_Cost_BEST=("Forecasted_Cost_BEST", "sum")).reset_index()
        
        zones_gdf_copy = self.zones_gdf.rename(columns={"ZoneName": "Zone_Name"})
        zones_gdf_copy["Zone_Name"] = zones_gdf_copy["Zone_Name"].str.strip().str.upper()
        zone_forecast["Zone_Name"] = zone_forecast["Zone_Name"].str.strip().str.upper()
        merged_gdf = zones_gdf_copy.merge(zone_forecast, on="Zone_Name", how="left")

        fig_map, ax_map = plt.subplots(1, 1, figsize=(12, 10))
        merged_gdf.plot(
            column="Forecasted_Cost_BEST", cmap="YlOrRd", legend=True, ax=ax_map,
            edgecolor="black", missing_kwds={"color": "lightgrey"}
        )
        ax_map.set_title(f"Forecasted Project Cost by Zone ({self.best_model_name})", fontsize=16)
        ax_map.axis("off")
        
        map_path = os.path.join(artifacts_dir, "1c_cost_forecast_map.png")
        fig_map.savefig(map_path, dpi=300, bbox_inches='tight')
        plot_paths["cost_forecast_map"] = os.path.abspath(map_path)
        plt.close(fig_map)

        top_zones = self.processed_df["Zone_Name"].value_counts().nlargest(6).index
        fig_pies, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, zone in enumerate(top_zones):
            zone_data = self.processed_df[self.processed_df["Zone_Name"] == zone]
            type_counts = zone_data["Project_Type"].value_counts()
            axes[i].pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=140)
            axes[i].set_title(f"Project Type Distribution: {zone}", fontsize=14)
        
        for i in range(len(top_zones), len(axes)):
            axes[i].axis('off') 
            
        plt.tight_layout()
        pie_path = os.path.join(artifacts_dir, "1c_zone_project_distribution.png")
        fig_pies.savefig(pie_path, dpi=300, bbox_inches='tight')
        plot_paths["project_distribution_pies"] = os.path.abspath(pie_path)
        plt.close(fig_pies)
        
        print(f"[1C] Generated {len(plot_paths)} plots in {artifacts_dir}/")
        return plot_paths

    def run_analysis(self, save_path: str, data_save_path: str, model_save_path: str):
        print("[1C] Preprocessing data...")
        self.processed_df = self._preprocess_data(self.projects_df, self.market_df)
        
        features = ["Demand_Index", "Project_Type"]
        target = "Planned_Cost_INR"
        X = self.processed_df[features]
        y = self.processed_df[target]

        print("[1C] Training scratch models...")
        scratch_metrics = self._train_scratch_models(X.drop('Project_Type', axis=1), y)
        
        print("[1C] Training inbuilt models...")
        inbuilt_metrics = self._train_inbuilt_models(X, y)
        
        if inbuilt_metrics["RandomForest"]["r2"] >= inbuilt_metrics["LinearRegression"]["r2"]:
            self.best_model = self.models["RandomForest"]
            self.best_model_name = "RandomForest"
        else:
            self.best_model = self.models["LinearRegression"]
            self.best_model_name = "LinearRegression"
        
        print(f"[1C] Best model selected: {self.best_model_name}")
        self.processed_df["Forecasted_Cost_BEST"] = self.best_model.predict(X)
        
        print("[1C] Saving artifacts...")
        artifacts_dir = os.path.dirname(save_path)
        
        joblib.dump(self.best_model, model_save_path)
        print(f"[1C] Best model saved at: {os.path.abspath(model_save_path)}")
        
        self.processed_df.to_pickle(data_save_path)
        print(f"[1C] Processed data saved at: {os.path.abspath(data_save_path)}")

        print("[1C] Generating plots...")
        plot_paths = self._generate_plots(artifacts_dir)
        
        zone_dist = self.processed_df.groupby('Zone_Name')['Project_Type'].value_counts(normalize=True).unstack(fill_value=0).to_dict('index')

        self.results = {
            "metrics": {"scratch": scratch_metrics, "inbuilt": inbuilt_metrics},
            "best_model_name": self.best_model_name,
            "plot_paths": plot_paths,
            "zone_project_distribution": zone_dist,
            "forecast_summary": self.processed_df[["Zone_Name", "Project_Type", "Planned_Cost_INR", "Forecasted_Cost_BEST"]].head().to_dict('records')
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(self.results, f)
        print(f"[1C] Analysis results saved at: {os.path.abspath(save_path)}")
        
        return self.results

def execute_1c():
    BASE_DIR = r"C:\new\BusinessExpansionAndAcceleration"
    
    RAW_DIR = os.path.join(BASE_DIR, "models")
    ZONE_DIR = os.path.join(BASE_DIR, "models", "Zone", "ZONE.shp")
    
    OUTPUT_DIR = os.path.join(BASE_DIR, "models", "artifacts")
    
    MODEL_PATH = os.path.join(OUTPUT_DIR, "1c_cost_model.pkl")
    RESULTS_PATH = os.path.join(OUTPUT_DIR, "1c_results.pkl")
    DATA_PATH = os.path.join(OUTPUT_DIR, "1c_forecast_output.pkl")

    print("--- Running Cost Forecasting Pipeline (1c) ---")
    print(f"  Input Dir (CSVs): {RAW_DIR}")
    print(f"  Output Dir: {OUTPUT_DIR}")
    
    pipeline = CostForecastingPipeline(
        raw_dir=RAW_DIR,
        zone_dir=ZONE_DIR
    )
    
    results = pipeline.run_analysis(
        save_path=RESULTS_PATH,
        data_save_path=DATA_PATH,
        model_save_path=MODEL_PATH
    )
    
    return results

if __name__ == "__main__":
    results = execute_1c()
    
    print("\n[+] Pipeline execution complete.")

    print("\n" + "="*50)
    print("--- PIPELINE 1C: COST FORECASTING COMPLETE ---")
    print("="*50 + "\n")
    
    print("--- Model Evaluation Metrics ---")
    print("\nScratch Models:")
    print(f"  Linear Regression → MSE: {results['metrics']['scratch']['LinearRegression']['mse']:.2f}, R2: {results['metrics']['scratch']['LinearRegression']['r2']:.3f}")
    print(f"  Random Forest   → MSE: {results['metrics']['scratch']['RandomForest']['mse']:.2f}, R2: {results['metrics']['scratch']['RandomForest']['r2']:.3f} (Not Implemented)")
    
    print("\nInbuilt Models:")
    print(f"  Linear Regression    → MSE: {results['metrics']['inbuilt']['LinearRegression']['mse']:.2f}, R2: {results['metrics']['inbuilt']['LinearRegression']['r2']:.3f}")
    print(f"  Random Forest Reg. → MSE: {results['metrics']['inbuilt']['RandomForest']['mse']:.2f}, R2: {results['metrics']['inbuilt']['RandomForest']['r2']:.3f}")

    print(f"\nBest Model: {results['best_model_name']}")
    
    print("\n--- Final Data with Forecasts (Top 5 Rows) ---")
    print(pd.DataFrame(results['forecast_summary']).to_string())
    
    print("\n--- Generated Plot Paths ---")
    for fig_name, fig_path in results['plot_paths'].items():
        print(f"  - Figure '{fig_name}' saved at: {fig_path}")

    print("\n--- End of Pipeline ---")