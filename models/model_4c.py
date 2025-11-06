import os
import warnings
import pandas as pd
import numpy as np
import joblib
import pickle  # Added import
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, Tuple, List

# --- 1. Configuration and Setup ---
warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Use paths from your second script
BASE_DIR = r"C:\MLOps_Project"
RAW_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = r"C:\new\BusinessExpansionAndAcceleration\models\artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define full output paths (from your first script's logic)
RESULTS_PATH = os.path.join(OUTPUT_DIR, "4c_results.pkl")
DATA_PATH = os.path.join(OUTPUT_DIR, "4c_processed_output.pkl")
MODEL_PATH = os.path.join(OUTPUT_DIR, "4c.pkl")


# --- 2. Helper Functions ---
def _normalize_competitor_id(cid):
    """Standardizes competitor IDs to COMP_XXX format."""
    if pd.isna(cid): return np.nan
    match = re.search(r'(\d+)', str(cid))
    return f"COMP_{int(match.group(1)):03d}" if match else cid

# --- 3. Data Loading and Preprocessing ---
def load_and_merge_data(raw_dir: str, output_dir: str) -> pd.DataFrame:
    """Loads, standardizes IDs, and merges all required data sources."""
    print("[+] Loading and merging data...")
    bids = pd.read_csv(os.path.join(raw_dir, "bids_history.csv"))
    projects = pd.read_csv(os.path.join(raw_dir, "competitor_projects.csv"))
    tenders = pd.read_csv(os.path.join(raw_dir, "tender.csv"))

    bids['Competitor_ID'] = bids['Competitor_ID'].apply(_normalize_competitor_id)
    projects['Competitor_ID'] = projects['Competitor_ID'].apply(_normalize_competitor_id)

    try:
        cost_forecast = pd.read_pickle(os.path.join(output_dir, "1c_forecast_output.pkl"))
        print("[+] Loaded 1c_forecast_output.pkl")
    except FileNotFoundError:
        print("[+] Warning: 1c_forecast_output.pkl not found. Proceeding without it.")
        cost_forecast = pd.DataFrame()
        
    try:
        # Use robust logic from Script 1
        segments_model = joblib.load(os.path.join(output_dir, "1d_clustering.pkl"))
        unique_competitors = projects['Competitor_ID'].unique()
        n_labels = len(segments_model.labels_)
        n_competitors = len(unique_competitors)
        
        if n_competitors >= n_labels:
            competitors_to_map = unique_competitors[:n_labels]
        else:
            competitors_to_map = list(unique_competitors) + [np.nan] * (n_labels - n_competitors)

        segments = pd.DataFrame({
            "Competitor_ID": competitors_to_map,
            "Segment_Name": segments_model.labels_
        })
        segments['Competitor_ID'] = segments['Competitor_ID'].apply(_normalize_competitor_id)
        print("[+] Loaded 1d_clustering.pkl and created segments.")
    except (FileNotFoundError, AttributeError, EOFError, ValueError) as e:
        print(f"[+] Warning: 1d_clustering.pkl not found or invalid ({e}). Proceeding without segments.")
        segments = pd.DataFrame()

    df = pd.merge(bids, tenders, on="Tender_ID", how="left")
    df = pd.merge(df, projects[['Competitor_ID', 'Project_Type', 'Project_Size_sq_ft', 'Pricing_Strategy']], on="Competitor_ID", how="left")
    
    if not cost_forecast.empty and "Project_ID" in cost_forecast.columns:
        df = pd.merge(df, cost_forecast, left_on="Tender_ID", right_on="Project_ID", how="left", suffixes=('', '_cost'))
    
    if not segments.empty:
        df = pd.merge(df, segments[['Competitor_ID', 'Segment_Name']], on="Competitor_ID", how="left")
        
    print(f"[+] Data loading and merging complete. {len(df)} rows loaded.")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Adds new features based on bid and cost data."""
    print("[+] Running feature engineering...")
    df["Bid_Amount_INR"] = pd.to_numeric(df["Bid_Amount_INR"], errors="coerce")
    df["Estimated_Cost_INR"] = pd.to_numeric(df["Estimated_Cost_INR"], errors="coerce")
    df["Discount_vs_Estimate"] = (df["Estimated_Cost_INR"] - df["Bid_Amount_INR"]) / df["Estimated_Cost_INR"].replace(0, np.nan)
    df["Bid_Success"] = (df["Win_Loss"] == "Win").astype(int)
    return df


# --- 4. Model Training and Evaluation ---
def train_scratch_models(X_train, X_test, y_train, y_test) -> Dict[str, Dict[str, float]]:
    """Trains and evaluates a scratch logistic regression for binary cases."""
    if len(np.unique(y_train)) != 2:
        print("[+] Scratch Model: Skipped (target is not binary).")
        return {"Logistic Regression": {"F1-Score": 0.0}}

    def sigmoid(z): return 1 / (1 + np.exp(-z))
    
    X_train_lr = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_lr = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    weights = np.zeros(X_train_lr.shape[1])

    for _ in range(1000):
        preds = sigmoid(X_train_lr @ weights)
        gradient = X_train_lr.T @ (y_train - preds) / len(y_train)
        weights += 0.01 * gradient

    preds_lr = (sigmoid(X_test_lr @ weights) >= 0.5).astype(int)
    f1 = f1_score(y_test, preds_lr, zero_division=0)
    
    return {"Logistic Regression": {"F1-Score": f1}}

def train_inbuilt_models(X_train, X_test, y_train, y_test) -> Tuple[Any, str, Dict]:
    """Trains inbuilt models and returns the best one along with all metrics."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42), # Matched Script 1
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }

    results = {}
    best_model, best_score = None, -1

    print("[+] Training inbuilt models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        results[name] = {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, preds, average='weighted', zero_division=0),
            "F1-Score": f1,
        }
        if f1 > best_score:
            best_score, best_model = f1, model

    best_model_name = best_model.__class__.__name__
    return best_model, best_model_name, results


# --- 5. Visualization ---
def generate_plots(artifacts_dir: str, processed_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, le: LabelEncoder) -> Dict[str, str]:
    """Generates and saves all required plots."""
    print(f"[+] Generating visualizations...")
    os.makedirs(artifacts_dir, exist_ok=True)
    plot_paths = {}
    sns.set_style("whitegrid")
    
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=le.transform(le.classes_))
    fig1, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set(title=f'Confusion Matrix - {model_name}', xlabel="Predicted", ylabel="Actual")
    path = os.path.join(artifacts_dir, "4c_confusion_matrix.png")
    fig1.savefig(path, dpi=300, bbox_inches='tight')
    plot_paths['confusion_matrix'] = os.path.abspath(path)
    plt.close(fig1)

    # Plot 2: Strategy Distribution
    fig2, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=processed_df['Predicted_Strategy'], ax=ax, palette='viridis', order=sorted(processed_df['Predicted_Strategy'].unique()))
    ax.set_title("Distribution of Predicted Pricing Strategies")
    ax.tick_params(axis='x', rotation=45)
    path = os.path.join(artifacts_dir, "4c_strategy_distribution.png")
    fig2.savefig(path, dpi=300, bbox_inches='tight')
    plot_paths['strategy_distribution'] = os.path.abspath(path)
    plt.close(fig2)
    
    print(f"[+] Generated {len(plot_paths)} plots in {artifacts_dir}/")
    return plot_paths

# --- 6. Main Pipeline Orchestrator ---
def run_pipeline_4c(raw_dir: str, output_dir: str, save_path: str, data_save_path: str, model_save_path: str):
    """Runs the entire Pricing Strategy Prediction pipeline."""
    print("--- Running Pricing Strategy Prediction Pipeline (4c) ---")
    print(f"   Input Dir (CSVs): {raw_dir}")
    print(f"   Artifacts Dir (In/Out): {output_dir}")

    df = load_and_merge_data(raw_dir, output_dir)
    processed_df = feature_engineering(df)
    
    processed_df.dropna(subset=['Pricing_Strategy'], inplace=True)
    if processed_df.empty:
        print("[!] ERROR: No data with 'Pricing_Strategy' available after merging. Aborting.")
        return {}
        
    feature_cols = ["Project_Type", "Segment_Name", "Project_Size_sq_ft", "Demand_Index"]
    for col in feature_cols:
        if col not in processed_df.columns:
            processed_df[col] = 'Missing'
            
    X = pd.get_dummies(processed_df[feature_cols], dummy_na=True, drop_first=True).fillna(0)
    
    le = LabelEncoder()
    y = le.fit_transform(processed_df["Pricing_Strategy"])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scratch_metrics = train_scratch_models(X_train.values, X_test.values, y_train, y_test)
    
    best_model, best_model_name, inbuilt_metrics = train_inbuilt_models(X_train, X_test, y_train, y_test)
    
    # Retrain best model on full data
    best_model.fit(X, y)
    processed_df['Predicted_Strategy'] = le.inverse_transform(best_model.predict(X))
    
    print("[+] Saving artifacts...")
    
    joblib.dump(best_model, model_save_path)
    print(f"[+] Best model saved at: {os.path.abspath(model_save_path)}")
    
    processed_df.to_pickle(data_save_path)
    print(f"[+] Processed data saved at: {os.path.abspath(data_save_path)}")

    plot_paths = generate_plots(output_dir, processed_df, y, best_model.predict(X), best_model_name, le)
    
    results = {
        "metrics": {"scratch": scratch_metrics, "inbuilt": inbuilt_metrics},
        "best_model_name": best_model_name,
        "plot_paths": plot_paths,
        "strategy_distribution_top5": processed_df['Predicted_Strategy'].value_counts().head(5).to_dict()
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[+] Analysis results saved at: {os.path.abspath(save_path)}")
    
    return results

if __name__ == "__main__":
    results = run_pipeline_4c(
        raw_dir=RAW_DIR,
        output_dir=OUTPUT_DIR,
        save_path=RESULTS_PATH,
        data_save_path=DATA_PATH,
        model_save_path=MODEL_PATH
    )
    
    if results:
        print("\n[+] Pipeline execution complete.")
        print("\n" + "="*50 + "\n--- PIPELINE 4C: PRICING STRATEGY COMPLETE ---\n" + "="*50)
        
        print("\n--- Scratch Models ---")
        for name, m in results['metrics']['scratch'].items():
            print(f"   {name}: F1-Score={m['F1-Score']:.4f}")
            
        print("\n--- Inbuilt Models ---")
        for name, m in results['metrics']['inbuilt'].items():
            print(f"   {name}: F1-Score={m['F1-Score']:.4f}")

        print(f"\nBest Model: {results['best_model_name']}")
        
        print("\n--- Distribution of Predicted Strategies (Top 5) ---")
        for strategy, count in results['strategy_distribution_top5'].items():
            print(f"   - {strategy}: {count}")
        
        print("\n--- Generated Plot Paths ---")
        for fig_name, fig_path in results['plot_paths'].items():
            print(f"   - Figure '{fig_name}' saved at: {fig_path}")
    else:
        print("\n[!] Pipeline execution failed (e.g., no data).")

    print("\n--- End of Pipeline ---")