import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, Any, Tuple, List

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

class AssetROIPipeline:

    def __init__(self, raw_dir: str):
        print(f"[4B] Initializing pipeline and loading data...")
        self.raw_dir = raw_dir
        
        try:
            self.assets_df, self.financials_df = self._load_data(raw_dir)
            print(f"[4B] Loaded {len(self.assets_df)} asset records and {len(self.financials_df)} financial records.")
        except FileNotFoundError as e:
            print(f"[4B] ERROR: Could not load input files.")
            print(f"[4B] Full error: {e}")
            raise
            
        self.processed_df = None
        self.models = {"best_model": None, "best_model_name": ""}
        self.metrics = {"scratch": {}, "inbuilt": {}}
        self.plot_data = {} # To store X_test, y_test for plotting
        self.results = {}

    def _load_data(self, raw_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        assets_df = pd.read_csv(os.path.join(raw_dir, "assets_performance.csv"))
        financials_df = pd.read_csv(os.path.join(raw_dir, "financials_revenue.csv"))
        return assets_df, financials_df

    def _preprocess_data(self, assets_df: pd.DataFrame) -> pd.DataFrame:
        print("[4B] Preprocessing data and creating performance scores...")
        df = assets_df.copy()
        df['Revenue_to_Cost'] = df['Revenue_Generated_INR'] / df['Investment_Cost_INR'].replace(0, 1)
        
        performance_features = ['ROI_Percent', 'Utilization_Rate_Percent', 'Revenue_to_Cost']
        df.dropna(subset=performance_features, inplace=True)

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df[performance_features])
        df['Performance_Score'] = scaled_features.mean(axis=1)
        
        df['Performance_Flag'] = pd.cut(
            df['Performance_Score'],
            bins=[-np.inf, 0.4, 0.7, np.inf],
            labels=["Underperforming", "Moderate", "High Performing"]
        )
        return df

    def _train_scratch_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        print("[4B] Training models from scratch...")
        
        def decision_stump(X_stump, y_stump):
            best_mse, best_criteria = float("inf"), None
            for f in range(X_stump.shape[1]):
                for t in np.unique(X_stump[:, f]):
                    left_mask = X_stump[:, f] <= t
                    if left_mask.sum() > 0 and (~left_mask).sum() > 0:
                        mse = np.mean((y_stump[left_mask] - y_stump[left_mask].mean())**2) + np.mean((y_stump[~left_mask] - y_stump[~left_mask].mean())**2)
                        if mse < best_mse:
                            best_mse, best_criteria = mse, (f, t)
            if best_criteria is None: # Handle case where no split improves
                return (0, np.mean(X_stump[:, 0]))
            return best_criteria

        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        preds_lr = X_b @ theta

        trees_rf = [decision_stump(X, y) for _ in range(10)]
        preds_rf = np.mean([
            np.where(X[:, tree[0]] <= tree[1], y[X[:, tree[0]] <= tree[1]].mean(), y[X[:, tree[0]] > tree[1]].mean())
            for tree in trees_rf
        ], axis=0)

        preds_gb = np.zeros_like(y, dtype=float)
        residual = y.copy().astype(float)
        for _ in range(10):
            tree = decision_stump(X, residual)
            pred = np.where(X[:, tree[0]] <= tree[1], residual[X[:, tree[0]] <= tree[1]].mean(), residual[X[:, tree[0]] > tree[1]].mean())
            preds_gb += 0.1 * pred
            residual -= 0.1 * pred

        results = {}
        for name, preds in [("Linear Regression", preds_lr), ("Random Forest", preds_rf), ("Gradient Boosting", preds_gb)]:
            results[name] = {"R2": r2_score(y, preds), "RMSE": np.sqrt(mean_squared_error(y, preds))}
            
        return results

    def _train_inbuilt_models(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, str, Dict[str, Any]]:
        print("[4B] Training inbuilt scikit-learn models...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42)
        }
        
        metrics, best_r2, best_model_name = {}, -np.inf, ""
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            r2, rmse = r2_score(y_test, preds), np.sqrt(mean_squared_error(y_test, preds))
            metrics[name] = {"R2": r2, "RMSE": rmse}
            if r2 > best_r2:
                best_r2, best_model_name = r2, name

        best_model = models[best_model_name]
        best_model.fit(X, y)
        
        self.plot_data = {"X_test": X_test, "y_test": y_test, "y_pred": best_model.predict(X_test)}
        
        return best_model, best_model_name, metrics

    def _generate_plots(self, artifacts_dir: str) -> Dict[str, str]:
        print(f"[4B] Generating visualizations...")
        os.makedirs(artifacts_dir, exist_ok=True)
        plot_paths = {}
        sns.set_style("whitegrid")
        df = self.processed_df
        
        # Plot 1: ROI by Performance
        fig1, ax = plt.subplots(figsize=(10, 6))
        # Plot only top 50 assets for readability
        top_assets = df.nlargest(50, 'ROI_Percent')
        sns.barplot(x='Asset_ID', y='ROI_Percent', hue='Performance_Flag', data=top_assets, ax=ax, dodge=False)
        ax.set_title("Asset ROI by Performance Flag (Top 50 Assets)")
        ax.tick_params(axis='x', rotation=90)
        path = os.path.join(artifacts_dir, "4b_roi_by_performance.png")
        fig1.savefig(path, dpi=300, bbox_inches='tight')
        plot_paths['roi_by_performance'] = os.path.abspath(path)
        plt.close(fig1)

        # Plot 2: Utilization vs ROI
        fig2, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='Utilization_Rate_Percent', y='ROI_Percent', hue='Performance_Flag', data=df, s=100, ax=ax)
        ax.set_title("Utilization Rate vs. ROI")
        path = os.path.join(artifacts_dir, "4b_utilization_vs_roi.png")
        fig2.savefig(path, dpi=300, bbox_inches='tight')
        plot_paths['utilization_vs_roi'] = os.path.abspath(path)
        plt.close(fig2)

        # Plot 3: Actual vs Predicted
        y_test = self.plot_data['y_test']
        y_pred = self.plot_data['y_pred']
        model_name = self.models['best_model_name']
        
        fig3, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set(xlabel="Actual ROI", ylabel="Predicted ROI", title=f"Predicted vs. Actual ROI ({model_name})")
        ax.grid(True)
        path = os.path.join(artifacts_dir, "4b_actual_vs_predicted.png")
        fig3.savefig(path, dpi=300, bbox_inches='tight')
        plot_paths['actual_vs_predicted'] = os.path.abspath(path)
        plt.close(fig3)
        
        print(f"[4B] Generated {len(plot_paths)} plots in {artifacts_dir}/")
        return plot_paths

    def run_analysis(self, save_path: str, data_save_path: str, model_save_path: str):
        print("[4B] Running full analysis pipeline...")
        self.processed_df = self._preprocess_data(self.assets_df)
        
        features = ['Investment_Cost_INR', 'Revenue_Generated_INR', 'Utilization_Rate_Percent', 'Revenue_to_Cost']
        target = 'ROI_Percent'
        
        # Ensure processed_df is not empty after preprocessing/dropping NaNs
        if self.processed_df.empty:
            print("[4B] ERROR: No data remaining after preprocessing. Aborting.")
            return {}

        X, y = self.processed_df[features], self.processed_df[target]
        
        self.metrics["scratch"] = self._train_scratch_models(X.values, y.values)
        
        best_model, best_model_name, inbuilt_metrics = self._train_inbuilt_models(X, y)
        self.models["best_model"] = best_model
        self.models["best_model_name"] = best_model_name
        self.metrics["inbuilt"] = inbuilt_metrics
        
        print("[4B] Saving artifacts...")
        artifacts_dir = os.path.dirname(save_path)
        
        joblib.dump(self.models["best_model"], model_save_path)
        print(f"[4B] Best model saved at: {os.path.abspath(model_save_path)}")
        
        self.processed_df.to_pickle(data_save_path)
        print(f"[4B] Processed data saved at: {os.path.abspath(data_save_path)}")

        plot_paths = self._generate_plots(artifacts_dir)
        
        self.results = {
            "metrics": self.metrics,
            "best_model_name": self.models["best_model_name"],
            "plot_paths": plot_paths,
            "processed_data_top5": self.processed_df[['Asset_ID', 'ROI_Percent', 'Performance_Score', 'Performance_Flag']].head().to_dict('records')
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(self.results, f)
        print(f"[4B] Analysis results saved at: {os.path.abspath(save_path)}")
        
        return self.results

def execute_4b():
    BASE_DIR = r"C:\new\BusinessExpansionAndAcceleration"
    
    RAW_DIR = os.path.join(BASE_DIR, "models")
    OUTPUT_DIR = os.path.join(BASE_DIR, "models", "artifacts")
    
    RESULTS_PATH = os.path.join(OUTPUT_DIR, "4b_results.pkl")
    DATA_PATH = os.path.join(OUTPUT_DIR, "4b_processed_output.pkl")
    MODEL_PATH = os.path.join(OUTPUT_DIR, "4b_roi_model.pkl")

    print("--- Running Asset ROI Prediction Pipeline (4b) ---")
    print(f"  Input Dir (CSVs): {RAW_DIR}")
    print(f"  Output Dir: {OUTPUT_DIR}")
    
    pipeline = AssetROIPipeline(raw_dir=RAW_DIR)
    
    results = pipeline.run_analysis(
        save_path=RESULTS_PATH,
        data_save_path=DATA_PATH,
        model_save_path=MODEL_PATH
    )
    
    return results

if __name__ == "__main__":
    results = execute_4b()
    
    if results:
        print("\n[+] Pipeline execution complete.")
        print("\n" + "="*50 + "\n--- PIPELINE 4B: ASSET ROI PREDICTION COMPLETE ---\n" + "="*50)
        
        print("\n--- Scratch Models ---")
        for name, m in results['metrics']['scratch'].items():
            print(f"  {name}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}")
            
        print("\n--- Inbuilt Models ---")
        for name, m in results['metrics']['inbuilt'].items():
            print(f"  {name}: R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}")

        print(f"\nBest Model: {results['best_model_name']}")
        
        print("\n--- Final Processed Data (Top 5 Rows) ---")
        print(pd.DataFrame(results['processed_data_top5']).to_string())
        
        print("\n--- Generated Plot Paths ---")
        for fig_name, fig_path in results['plot_paths'].items():
            print(f"  - Figure '{fig_name}' saved at: {fig_path}")
    else:
        print("\n[!] Pipeline execution failed (e.g., no data).")

    print("\n--- End of Pipeline ---")