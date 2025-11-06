import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
from typing import Dict, Any, Tuple, List

# --- 1. Configuration and Setup ---
warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

class ESGCompliancePipeline:

    def __init__(self, raw_dir: str):
        print(f"[5B] Initializing pipeline and loading data...")
        self.raw_dir = raw_dir
        
        try:
            self.raw_df = self._load_data(raw_dir)
            print(f"[5B] Loaded {len(self.raw_df)} ESG compliance records.")
        except FileNotFoundError as e:
            print(f"[5B] ERROR: Could not load input file.")
            print(f"[5B] Full error: {e}")
            raise
            
        self.processed_df = None
        self.models = {"best_model": None, "best_model_name": ""}
        self.metrics = {"scratch": {}, "inbuilt": {}}
        self.results = {}
        self.features = ['Safety_Incidents', 'Labor_Compliance', 'Wage_Compliance', 'Emission_Level']
        self.target = 'Compliance_Status'

    def _load_data(self, raw_dir: str) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(raw_dir, "esg_compliance.csv"))
        df.fillna("Unknown", inplace=True)
        return df

    def _train_scratch_model(self, df: pd.DataFrame, features: List[str], target: str) -> Dict[str, Any]:
        print("[5B] Training model from scratch...")
        X = df[features]
        y = df[target]

        X_enc = pd.get_dummies(X, columns=['Labor_Compliance', 'Wage_Compliance', 'Emission_Level'])
        X_values = X_enc.values.astype(float)
        y_values, class_names = pd.factorize(y)

        X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size=0.2, random_state=42)

        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_train))
        X_bias = np.c_[np.ones((n_samples, 1)), X_train]
        W = np.zeros((n_features + 1, n_classes))
        y_onehot = np.eye(n_classes)[y_train]

        for _ in range(1000):
            z = X_bias @ W
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            softmax = exp_z / exp_z.sum(axis=1, keepdims=True)
            grad = X_bias.T @ (softmax - y_onehot) / n_samples
            W -= 0.01 * grad

        X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        y_pred = np.argmax(X_test_bias @ W, axis=1)

        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1_Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        return {
            "metrics": {"LogisticRegression_Scratch": metrics},
            "confusion_matrix": pd.DataFrame(cm, index=class_names, columns=class_names)
        }

    def _train_inbuilt_models(self, df: pd.DataFrame, features: List[str], target: str) -> Tuple[Pipeline, str, Dict[str, Any]]:
        print("[5B] Training inbuilt scikit-learn models...")
        X, y = df[features], df[target]
        cat_features = ['Labor_Compliance', 'Wage_Compliance', 'Emission_Level']
        num_features = ['Safety_Incidents']

        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42), # Reduced
            "GradientBoosting": GradientBoostingClassifier(random_state=42)
        }

        results = {}
        metrics_data = {}
        best_pipeline, best_f1 = None, -1.0

        for name, model in models.items():
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            
            pipeline.fit(X, y)
            preds = pipeline.predict(X)
            
            f1 = f1_score(y, preds, average="weighted", zero_division=0)
            metrics_data[name] = {
                "Accuracy": accuracy_score(y, preds),
                "Precision": precision_score(y, preds, average="weighted", zero_division=0),
                "Recall": recall_score(y, preds, average="weighted", zero_division=0),
                "F1_Score": f1,
                "CV_Accuracy": cross_val_score(pipeline, X, y, cv=5, scoring="accuracy").mean()
            }
            
            if f1 > best_f1:
                best_f1, best_pipeline = f1, pipeline

        best_model_name = best_pipeline.named_steps['classifier'].__class__.__name__
        return best_pipeline, best_model_name, metrics_data

    def _analyze_risk(self, df: pd.DataFrame, pipeline: Pipeline, features: List[str]) -> pd.DataFrame:
        print("[5B] Analyzing risk scores...")
        df_copy = df.copy()
        
        if len(pipeline.classes_) == 2:
            df_copy['Risk_Score'] = pipeline.predict_proba(df_copy[features])[:, 1]
        else:
            df_copy['Risk_Score'] = np.max(pipeline.predict_proba(df_copy[features]), axis=1)
        
        df_copy['Risk_Level'] = pd.cut(df_copy['Risk_Score'], bins=[0, 0.33, 0.66, 1.01], labels=["Low", "Medium", "High"], right=False)
        return df_copy

    def _generate_plots(self, artifacts_dir: str, y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> Dict[str, str]:
        print(f"[5B] Generating visualizations...")
        os.makedirs(artifacts_dir, exist_ok=True)
        plot_paths = {}
        sns.set_style("whitegrid")

        cm = confusion_matrix(y_true, y_pred)
        classes = np.unique(y_true)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set(xlabel="Predicted Label", ylabel="True Label", title=f"Confusion Matrix ({model_name})")
        
        path = os.path.join(artifacts_dir, "5b_confusion_matrix.png")
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plot_paths['confusion_matrix'] = os.path.abspath(path)
        plt.close(fig)
        
        print(f"[5B] Generated {len(plot_paths)} plots in {artifacts_dir}/")
        return plot_paths

    def run_analysis(self, save_path: str, data_save_path: str, model_save_path: str):
        print("[5B] Running full analysis pipeline...")
        df = self.raw_df.copy()
        
        scratch_results = self._train_scratch_model(df, self.features, self.target)
        self.metrics["scratch"] = scratch_results["metrics"]
        self.metrics["scratch_cm"] = scratch_results["confusion_matrix"]
        
        best_pipeline, best_model_name, inbuilt_metrics = self._train_inbuilt_models(df, self.features, self.target)
        self.models["best_model"] = best_pipeline
        self.models["best_model_name"] = best_model_name
        self.metrics["inbuilt"] = inbuilt_metrics
        
        self.processed_df = self._analyze_risk(df, best_pipeline, self.features)
        
        y_pred_full = best_pipeline.predict(df[self.features])
        
        print("[5B] Saving artifacts...")
        artifacts_dir = os.path.dirname(save_path)
        
        joblib.dump(self.models["best_model"], model_save_path)
        print(f"[5B] Best model saved at: {os.path.abspath(model_save_path)}")
        
        self.processed_df.to_pickle(data_save_path)
        print(f"[5B] Processed data saved at: {os.path.abspath(data_save_path)}")

        plot_paths = self._generate_plots(artifacts_dir, df[self.target], y_pred_full, best_model_name)
        
        self.results = {
            "metrics": self.metrics,
            "best_model_name": self.models["best_model_name"],
            "plot_paths": plot_paths,
            "risk_summary_top5": self.processed_df[['Project_ID', 'Compliance_Status', 'Risk_Score', 'Risk_Level']].head().to_dict('records')
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(self.results, f)
        print(f"[5B] Analysis results saved at: {os.path.abspath(save_path)}")
        
        return self.results

def execute_5b():
    BASE_DIR = r"C:\new\BusinessExpansionAndAcceleration"
    
    RAW_DIR = os.path.join(BASE_DIR, "models")
    OUTPUT_DIR = os.path.join(BASE_DIR, "models", "artifacts")
    
    RESULTS_PATH = os.path.join(OUTPUT_DIR, "5b_results.pkl")
    DATA_PATH = os.path.join(OUTPUT_DIR, "5b_processed_output.pkl")
    MODEL_PATH = os.path.join(OUTPUT_DIR, "5b.pkl") # As requested

    print("--- Running ESG Compliance Pipeline (5b) ---")
    print(f"  Input Dir (CSVs): {RAW_DIR}")
    print(f"  Output Dir: {OUTPUT_DIR}")
    
    pipeline = ESGCompliancePipeline(raw_dir=RAW_DIR)
    
    results = pipeline.run_analysis(
        save_path=RESULTS_PATH,
        data_save_path=DATA_PATH,
        model_save_path=MODEL_PATH
    )
    
    return results

if __name__ == "__main__":
    results = execute_5b()
    
    if results:
        print("\n[+] Pipeline execution complete.")
        print("\n" + "="*50 + "\n--- PIPELINE 5B: ESG COMPLIANCE COMPLETE ---\n" + "="*50)
        
        print("\n--- Scratch Model → Logistic Regression ---")
        scratch_metrics_df = pd.DataFrame.from_dict(results['metrics']['scratch'], orient='index')
        print(scratch_metrics_df.round(4))
        print("\nConfusion Matrix:\n", results['metrics']['scratch_cm'])
            
        print("\n--- Inbuilt Models ---")
        inbuilt_metrics_df = pd.DataFrame.from_dict(results['metrics']['inbuilt'], orient='index')
        print(inbuilt_metrics_df.round(4))
              
        print(f"\nBest model: {results['best_model_name']}")
        
        print("\n--- Risk Score + Risk Level Table (Top 5 Rows) ---")
        print(pd.DataFrame(results['risk_summary_top5']).to_string())
        
        print("\n--- Generated Plot Paths ---")
        for fig_name, fig_path in results['plot_paths'].items():
            print(f"  - Figure '{fig_name}' saved at: {fig_path}")
    else:
        print("\n[!] Pipeline execution failed (e.g., no data).")

    print("\n--- End of Pipeline ---")