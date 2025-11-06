import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Tuple, List

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

class EmissionsPredictionPipeline:

    def __init__(self, raw_dir: str):
        print(f"[5A] Initializing pipeline and loading data...")
        self.raw_dir = raw_dir
        
        try:
            self.raw_df = self._load_data(raw_dir)
            print(f"[5A] Loaded {len(self.raw_df)} emission records.")
        except FileNotFoundError as e:
            print(f"[5A] ERROR: Could not load input file.")
            print(f"[5A] Full error: {e}")
            raise
            
        self.processed_df = None
        self.models = {"best_model": None, "best_model_name": ""}
        self.metrics = {"scratch": {}, "inbuilt": {}}
        self.plot_data = {}
        self.results = {}
        self.features = ['Cement_Usage_bags', 'Diesel_Usage_litres', 'Electricity_Usage_kWh', 'Project_Size_sq_ft']
        self.target = 'Emissions_CO2_tonnes'

    def _load_data(self, raw_dir: str) -> pd.DataFrame:
        file_path = os.path.join(raw_dir, "emissions_data.csv")
        df = pd.read_csv(file_path)
        df.fillna(0, inplace=True)
        return df

    def _train_scratch_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        print("[5A] Training scratch models...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Add bias term for normal equation
        X_train_b = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
        X_test_b = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]
        y_train_v = y_train.values
        y_test_v = y_test.values

        # --- Scratch Linear Regression (Normal Equation) ---
        try:
            theta_lr = np.linalg.pinv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train_v
            y_pred_lr = X_test_b @ theta_lr
        except np.linalg.LinAlgError:
            y_pred_lr = np.zeros_like(y_test_v)

        # --- Scratch Ridge Regression (Normal Equation) ---
        alpha = 1.0
        I = np.identity(X_train_b.shape[1])
        I[0, 0] = 0 # Don't regularize bias term
        try:
            theta_ridge = np.linalg.pinv(X_train_b.T @ X_train_b + alpha * I) @ X_train_b.T @ y_train_v
            y_pred_ridge = X_test_b @ theta_ridge
        except np.linalg.LinAlgError:
            y_pred_ridge = np.zeros_like(y_test_v)

        def get_metrics(y_true, y_pred):
            return {
                "R2": r2_score(y_true, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "MSE": mean_squared_error(y_true, y_pred),
                "MAE": mean_absolute_error(y_true, y_pred)
            }

        return {
            "Linear Regression": get_metrics(y_test_v, y_pred_lr),
            "Ridge Regression": get_metrics(y_test_v, y_pred_ridge)
        }

    def _train_inbuilt_models(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, str, Dict[str, Any]]:
        print("[5A] Training inbuilt scikit-learn models...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        metrics = {}
        best_r2 = -float('inf')
        best_model_name = ""
        
        # Pre-scale data for evaluation and CV
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_full_scaled = scaler.fit_transform(X) # For CV

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Get CV score
            cv_scores = cross_val_score(model, X_full_scaled, y, cv=5, scoring='r2')
            
            metrics[name] = {
                "R2": r2_score(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MSE": mean_squared_error(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "CV_R2": np.mean(cv_scores)
            }
            if metrics[name]["R2"] > best_r2:
                best_r2 = metrics[name]["R2"]
                best_model_name = name

        final_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', models[best_model_name])
        ])
        
        final_pipeline.fit(X, y)
        
        self.plot_data = {"X_test": X_test, "y_test": y_test, "y_pred": final_pipeline.predict(X_test)}
        
        return final_pipeline, best_model_name, metrics

    def _analyze_emissions(self, df: pd.DataFrame, pipeline: Pipeline, features: List[str]) -> pd.DataFrame:
        print("[5A] Analyzing emissions and flagging projects...")
        df['Predicted_Emissions'] = pipeline.predict(df[features])
        
        FACTOR_CEMENT = 0.06
        FACTOR_DIESEL = 0.0027
        FACTOR_ELEC = 0.0007

        df['Emissions_Cement'] = df['Cement_Usage_bags'] * FACTOR_CEMENT
        df['Emissions_Diesel'] = df['Diesel_Usage_litres'] * FACTOR_DIESEL
        df['Emissions_Electricity'] = df['Electricity_Usage_kWh'] * FACTOR_ELEC
        
        threshold = df['Predicted_Emissions'].mean() + df['Predicted_Emissions'].std()
        df['High_Emission_Flag'] = df['Predicted_Emissions'] > threshold
        
        return df

    def _generate_plots(self, artifacts_dir: str) -> Dict[str, str]:
        print(f"[5A] Generating visualizations...")
        os.makedirs(artifacts_dir, exist_ok=True)
        plot_paths = {}
        sns.set_style("whitegrid")

        y_test = self.plot_data['y_test']
        y_pred = self.plot_data['y_pred']
        model_name = self.models['best_model_name']

        # Plot 1: Actual vs Predicted
        fig1, ax1 = plt.subplots(figsize=(7, 6))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax1, alpha=0.8, edgecolor='k')
        ax1.axline((y_test.min(), y_test.min()), slope=1, color="red", linestyle="--")
        ax1.set_xlabel("Actual Emissions (tonnes)")
        ax1.set_ylabel("Predicted Emissions (tonnes)")
        ax1.set_title(f"Actual vs. Predicted Emissions ({model_name})")
        ax1.grid(True)
        path = os.path.join(artifacts_dir, "5a_actual_vs_predicted.png")
        fig1.savefig(path, dpi=300, bbox_inches='tight')
        plot_paths['actual_vs_predicted'] = os.path.abspath(path)
        plt.close(fig1)
        
        # Plot 2: Feature Importance
        model_object = self.models['best_model'].named_steps['regressor']
        if hasattr(model_object, "feature_importances_"):
            importances = model_object.feature_importances_
            feat_imp = pd.Series(importances, index=self.features).sort_values(ascending=False)
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax2, palette='viridis')
            ax2.set_title(f"Feature Importance ({model_name})")
            ax2.set_xlabel("Importance")
            path = os.path.join(artifacts_dir, "5a_feature_importance.png")
            fig2.savefig(path, dpi=300, bbox_inches='tight')
            plot_paths['feature_importance'] = os.path.abspath(path)
            plt.close(fig2)
        
        print(f"[5A] Generated {len(plot_paths)} plots in {artifacts_dir}/")
        return plot_paths

    def run_analysis(self, save_path: str, data_save_path: str, model_save_path: str):
        print("[5A] Running full analysis pipeline...")
        df = self.raw_df.copy()
        
        X = df[self.features]
        y = df[self.target]
        
        self.metrics["scratch"] = self._train_scratch_models(X, y)
        
        pipeline, best_model_name, inbuilt_metrics = self._train_inbuilt_models(X, y)
        self.models["best_model"] = pipeline
        self.models["best_model_name"] = best_model_name
        self.metrics["inbuilt"] = inbuilt_metrics
        
        self.processed_df = self._analyze_emissions(df, pipeline, self.features)
        
        print("[5A] Saving artifacts...")
        artifacts_dir = os.path.dirname(save_path)
        
        joblib.dump(self.models["best_model"], model_save_path)
        print(f"[5A] Best model saved at: {os.path.abspath(model_save_path)}")
        
        self.processed_df.to_pickle(data_save_path)
        print(f"[5A] Processed data saved at: {os.path.abspath(data_save_path)}")

        plot_paths = self._generate_plots(artifacts_dir)
        
        portfolio_actual = self.processed_df['Emissions_CO2_tonnes'].sum()
        portfolio_predicted = self.processed_df['Predicted_Emissions'].sum()

        self.results = {
            "metrics": self.metrics,
            "best_model_name": self.models["best_model_name"],
            "plot_paths": plot_paths,
            "portfolio_summary": {
                "Total Actual Emissions": portfolio_actual,
                "Total Predicted Emissions": portfolio_predicted
            },
            "processed_data_top5": self.processed_df[['Project_ID', 'Emissions_CO2_tonnes', 'Predicted_Emissions', 'High_Emission_Flag']].head().to_dict('records')
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(self.results, f)
        print(f"[5A] Analysis results saved at: {os.path.abspath(save_path)}")
        
        return self.results

def execute_5a():
    BASE_DIR = r"C:\new\BusinessExpansionAndAcceleration"
    
    RAW_DIR = os.path.join(BASE_DIR, "models")
    OUTPUT_DIR = os.path.join(BASE_DIR, "models", "artifacts")
    
    RESULTS_PATH = os.path.join(OUTPUT_DIR, "5a_results.pkl")
    DATA_PATH = os.path.join(OUTPUT_DIR, "5a_processed_output.pkl")
    MODEL_PATH = os.path.join(OUTPUT_DIR, "5a.pkl") # As requested

    print("--- Running Emissions Prediction Pipeline (5a) ---")
    print(f"  Input Dir (CSVs): {RAW_DIR}")
    print(f"  Output Dir: {OUTPUT_DIR}")
    
    pipeline = EmissionsPredictionPipeline(raw_dir=RAW_DIR)
    
    results = pipeline.run_analysis(
        save_path=RESULTS_PATH,
        data_save_path=DATA_PATH,
        model_save_path=MODEL_PATH
    )
    
    return results

if __name__ == "__main__":
    results = execute_5a()
    
    if results:
        print("\n[+] Pipeline execution complete.")
        print("\n" + "="*50 + "\n--- PIPELINE 5A: EMISSIONS PREDICTION COMPLETE ---\n" + "="*50)
        
        print("\n--- Scratch Models ---")
        for name, m in results['metrics']['scratch'].items():
            print(f"  {name}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MSE={m['MSE']:.4f}, MAE={m['MAE']:.4f}")
            
        print("\n--- Inbuilt Models ---")
        for name, m in results['metrics']['inbuilt'].items():
            print(f"  {name}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MSE={m['MSE']:.4f}, MAE={m['MAE']:.4f}, CV_R2={m['CV_R2']:.4f}")

        print(f"\nBest Model: {results['best_model_name']}")
        
        print("\n--- Portfolio Emission Summary ---")
        print(f"  Total Actual Emissions: {results['portfolio_summary']['Total Actual Emissions']:.2f} tonnes")
        print(f"  Total Predicted Emissions: {results['portfolio_summary']['Total Predicted Emissions']:.2f} tonnes")
        
        print("\n--- Final Data with Predictions (Top 5 Rows) ---")
        print(pd.DataFrame(results['processed_data_top5']).to_string())
        
        print("\n--- Generated Plot Paths ---")
        for fig_name, fig_path in results['plot_paths'].items():
            print(f"  - Figure '{fig_name}' saved at: {fig_path}")
    else:
        print("\n[!] Pipeline execution failed (e.g., no data).")

    print("\n--- End of Pipeline ---")