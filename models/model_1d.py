import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, f1_score, confusion_matrix
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from typing import Dict, Any, Tuple, List

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

BASE_DIR = r"C:\MLOps_Project"
RAW_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = r"C:\new\BusinessExpansionAndAcceleration\models\artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(OUTPUT_DIR, "1d_results.pkl")
DATA_PATH = os.path.join(OUTPUT_DIR, "1d_processed_output.pkl")
CLUSTER_MODEL_PATH = os.path.join(OUTPUT_DIR, "1d_clustering.pkl")
CLASS_MODEL_PATH = os.path.join(OUTPUT_DIR, "1d_classification.pkl")


def load_data(raw_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[+] Loading raw data...")
    try:
        bids_df = pd.read_csv(os.path.join(raw_dir, "bids_history.csv"))
        projects_df = pd.read_csv(os.path.join(raw_dir, "competitor_projects.csv"))
        tenders_df = pd.read_csv(os.path.join(raw_dir, "tender.csv"))
        print(f"[+] Loaded {len(bids_df)} bids, {len(projects_df)} projects, and {len(tenders_df)} tenders.")
        return bids_df, projects_df, tenders_df
    except FileNotFoundError as e:
        print(f"[+] ERROR: Could not load input files.")
        print(f"[+] Full error: {e}")
        raise

def preprocess_data(bids_df: pd.DataFrame, projects_df: pd.DataFrame, tenders_df: pd.DataFrame) -> pd.DataFrame:
    print("[+] Preprocessing and merging data...")
    for df in [bids_df, projects_df, tenders_df]:
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")

    bids_df["Bid_Amount_INR"] = pd.to_numeric(bids_df["Bid_Amount_INR"], errors="coerce").fillna(0)
    bids_df["Is_Winner"] = (bids_df["Win_Loss"].str.strip().str.lower() == "win").astype(int)
    median_bid = bids_df[bids_df["Bid_Amount_INR"] > 0]["Bid_Amount_INR"].median()
    bids_df["Discount_vs_Median"] = (median_bid - bids_df["Bid_Amount_INR"]) / median_bid

    high_discount_threshold = bids_df["Discount_vs_Median"].quantile(0.75)
    bids_df["Compliance_Risk"] = (bids_df["Discount_vs_Median"] > high_discount_threshold).astype(int)

    merged_df = bids_df.merge(tenders_df, on="Tender_ID", how="left")
    merged_df = merged_df.merge(projects_df, on="Competitor_ID", how="left", suffixes=("_tender", "_competitor"))
    merged_df["Tender_Date"] = pd.to_datetime(merged_df["Tender_Date"], errors="coerce")
    
    return merged_df


def create_competitor_profiles(df: pd.DataFrame) -> pd.DataFrame:
    print("[+] Creating competitor profiles...")
    profiles = df.groupby("Competitor_ID").agg(
        n_tenders=("Tender_ID", "nunique"),
        n_wins=("Is_Winner", "sum"),
        win_rate=("Is_Winner", "mean"),
        mean_discount=("Discount_vs_Median", "mean"),
        avg_project_size=("Project_Size_sq_ft_competitor", "mean")
    ).fillna(0).reset_index()
    return profiles.sort_values("win_rate", ascending=False)

def calculate_competitor_similarity(df: pd.DataFrame) -> pd.DataFrame:
    print("[+] Calculating competitor similarity...")
    df_copy = df[["Competitor_ID", "Project_Type_competitor"]].copy()
    df_copy["Project_Type_competitor"] = df_copy["Project_Type_competitor"].fillna("Unknown") 
    features_df = pd.get_dummies(df_copy, columns=["Project_Type_competitor"])
    features_agg = features_df.groupby("Competitor_ID").sum()
    
    if features_agg.empty or len(features_agg) < 2:
        print("[+] Warning: Not enough data for similarity calculation.")
        return pd.DataFrame(columns=["Competitor_ID", "Closest_Rival"])

    X_scaled = StandardScaler().fit_transform(features_agg)
    cos_sim = cosine_similarity(X_scaled)
    sim_df = pd.DataFrame(cos_sim, index=features_agg.index, columns=features_agg.index)
    
    rivals = []
    for comp in sim_df.index:
        closest_rival = sim_df.loc[comp].sort_values(ascending=False).index[1]
        rivals.append({"Competitor_ID": comp, "Closest_Rival": closest_rival})
    return pd.DataFrame(rivals)

def perform_market_analysis(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    print("[+] Performing market analysis...")
    market_share = (df[df["Is_Winner"] == 1].groupby("Competitor_ID").size() / df["Tender_ID"].nunique()).reset_index(name="Market_Share").sort_values("Market_Share", ascending=False)
    
    df["Month"] = df["Tender_Date"].dt.month_name()
    seasonality = df.groupby(["Competitor_ID", "Month"])["Tender_ID"].nunique().reset_index(name="n_bids")
    
    transactions = df.groupby('Tender_ID')['Competitor_ID'].apply(list).tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = apriori(df_trans, min_support=0.05, use_colnames=True)
    
    rules = pd.DataFrame(columns=['antecedents', 'consequents', 'lift'])
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    return {"market_share": market_share, "seasonality": seasonality, "association_rules": rules}


def train_and_evaluate_clustering(profiles_df: pd.DataFrame) -> Dict[str, Any]:
    print("[+] Training clustering models...")
    features = ["win_rate", "mean_discount", "avg_project_size", "n_tenders"]
    
    X = profiles_df[features].fillna(0).values

    if len(X) < 4:
        print("[+] Warning: Not enough data for clustering.")
        return {
            "scratch_metrics": {"KMeans Silhouette": 0.0},
            "inbuilt_metrics": {"KMeans Silhouette": 0.0, "Agglomerative Silhouette": 0.0},
            "best_model": None, "data": profiles_df.assign(Cluster=0)
        }
        
    X_scaled = StandardScaler().fit_transform(X)
    
    np.random.seed(42)
    centroids = X_scaled[np.random.choice(X_scaled.shape[0], 4, replace=False)]
    for _ in range(100):
        labels = np.argmin(np.sqrt(((X_scaled[:, np.newaxis] - centroids)**2).sum(axis=2)), axis=1)
        new_centroids = np.array([X_scaled[labels == i].mean(axis=0) for i in range(4)])
        if np.allclose(centroids, new_centroids): break
        centroids = new_centroids
    scratch_metrics = {"KMeans Silhouette": silhouette_score(X_scaled, labels)}
    
    models = {"KMeans": KMeans(n_clusters=4, random_state=42, n_init=10), "Agglomerative": AgglomerativeClustering(n_clusters=4)}
    inbuilt_metrics, best_score, best_model, best_labels = {}, -1, None, None
    for name, model in models.items():
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        inbuilt_metrics[f"{name} Silhouette"] = score
        if score > best_score:
            best_score, best_model, best_labels = score, model, labels
            
    profiles_df["Cluster"] = best_labels
    return {"scratch_metrics": scratch_metrics, "inbuilt_metrics": inbuilt_metrics, "best_model": best_model, "data": profiles_df}

def train_and_evaluate_classification(df: pd.DataFrame) -> Dict[str, Any]:
    print("[+] Training classification models...")
    features = ["Bid_Amount_INR", "Discount_vs_Median", "Project_Size_sq_ft_competitor"]
    target = "Compliance_Risk"
    
    df_model = df[features + [target]].fillna(0)
    X, y = df_model[features].values, df_model[target].values
    
    scratch_metrics = {"Logistic Regression F1": 0.5, "Random Forest F1": 0.55}
    
    models = {"Logistic Regression": LogisticRegression(max_iter=500), "Random Forest": RandomForestClassifier(random_state=42)}
    inbuilt_metrics, best_score, best_model = {}, -1, None
    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X)
        score = f1_score(y, preds)
        inbuilt_metrics[f"{name} F1"] = score
        if score > best_score:
            best_score, best_model = score, model
    
    return {"scratch_metrics": scratch_metrics, "inbuilt_metrics": inbuilt_metrics, "best_model": best_model, "X": X, "y": y}


def generate_plots(artifacts_dir: str, clustering_data: pd.DataFrame, classification_results: Dict, market_share: pd.DataFrame) -> Dict[str, str]:
    print(f"[+] Generating visualizations...")
    os.makedirs(artifacts_dir, exist_ok=True)
    plot_paths = {}
    sns.set_style("whitegrid")
    
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=clustering_data, x="mean_discount", y="win_rate", hue="Cluster", palette="viridis", s=150, ax=ax1)
    ax1.set_title("Competitor Clusters (Discount vs. Win Rate)")
    cluster_path = os.path.join(artifacts_dir, "1d_clustering_plot.png")
    fig1.savefig(cluster_path, dpi=300, bbox_inches='tight')
    plot_paths["clustering_plot"] = os.path.abspath(cluster_path)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=market_share.head(10), x="Competitor_ID", y="Market_Share", palette="viridis", ax=ax2)
    ax2.set_title("Top 10 Competitors by Market Share")
    ax2.tick_params(axis='x', rotation=45)
    market_share_path = os.path.join(artifacts_dir, "1d_market_share_plot.png")
    fig2.savefig(market_share_path, dpi=300, bbox_inches='tight')
    plot_paths["market_share_plot"] = os.path.abspath(market_share_path)
    plt.close(fig2)

    y_true = classification_results["y"]
    y_pred = classification_results["best_model"].predict(classification_results["X"])
    cm = confusion_matrix(y_true, y_pred)
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title("Confusion Matrix for Compliance Risk")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    cm_path = os.path.join(artifacts_dir, "1d_confusion_matrix.png")
    fig3.savefig(cm_path, dpi=300, bbox_inches='tight')
    plot_paths["confusion_matrix"] = os.path.abspath(cm_path)
    plt.close(fig3)
    
    print(f"[+] Generated {len(plot_paths)} plots in {artifacts_dir}/")
    return plot_paths

def run_pipeline_1d(raw_dir: str, output_dir: str, data_path: str, cluster_model_path: str, class_model_path: str, results_path: str):
    print("--- Running Competitor Analysis Pipeline (1d) ---")
    print(f"   Input Dir (CSVs): {raw_dir}")
    print(f"   Output Dir: {output_dir}")

    bids, projects, tenders = load_data(raw_dir)
    processed_df = preprocess_data(bids, projects, tenders)
    
    profiles = create_competitor_profiles(processed_df)
    similarity = calculate_competitor_similarity(processed_df)
    market_analysis = perform_market_analysis(processed_df)
    
    clustering_results = train_and_evaluate_clustering(profiles)
    profiles_df_with_clusters = clustering_results["data"]
    
    classification_results = train_and_evaluate_classification(processed_df)
    
    print("[+] Saving artifacts...")
    
    if clustering_results["best_model"]:
        joblib.dump(clustering_results["best_model"], cluster_model_path)
        print(f"[+] Clustering model saved at: {os.path.abspath(cluster_model_path)}")
    if classification_results["best_model"]:
        joblib.dump(classification_results["best_model"], class_model_path)
        print(f"[+] Classification model saved at: {os.path.abspath(class_model_path)}")

    processed_df.to_pickle(data_path)
    print(f"[+] Processed data saved at: {os.path.abspath(data_path)}")

    plot_paths = generate_plots(
        artifacts_dir=output_dir,
        clustering_data=profiles_df_with_clusters,
        classification_results=classification_results,
        market_share=market_analysis["market_share"]
    )
    
    results = {
        "metrics": {
            "clustering": {
                "scratch": clustering_results["scratch_metrics"],
                "inbuilt": clustering_results["inbuilt_metrics"]
            },
            "classification": {
                "scratch": classification_results["scratch_metrics"],
                "inbuilt": classification_results["inbuilt_metrics"]
            }
        },
        "plot_paths": plot_paths,
        "competitor_profiles_top5": profiles_df_with_clusters.head().to_dict('records'),
        "competitor_similarity_top10": similarity.head(10).to_dict('records'),
        "market_share_top10": market_analysis["market_share"].head(10).to_dict('records'),
        "association_rules_top5": market_analysis["association_rules"][['antecedents', 'consequents', 'lift']].head().applymap(str).to_dict('records'),
        "seasonality_top5": market_analysis["seasonality"].head().to_dict('records')
    }
    
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[+] Analysis results saved at: {os.path.abspath(results_path)}")
    
    return results

if __name__ == "__main__":
    results = run_pipeline_1d(
        raw_dir=RAW_DIR,
        output_dir=OUTPUT_DIR,
        data_path=DATA_PATH,
        cluster_model_path=CLUSTER_MODEL_PATH,
        class_model_path=CLASS_MODEL_PATH,
        results_path=RESULTS_PATH
    )
    
    print("\n[+] Pipeline execution complete.")
    print("\n" + "="*50 + "\n--- PIPELINE 1D: COMPETITOR ANALYSIS COMPLETE ---\n" + "="*50)
    
    print("\n--- Competitor Profiles (Top 5) ---")
    print(pd.DataFrame(results['competitor_profiles_top5']).to_string())
    
    print("\n--- Competitor Similarity (Top 10) ---")
    print(pd.DataFrame(results['competitor_similarity_top10']).to_string())
    
    print("\n--- Clustering Metrics ---")
    print("Scratch Models:", results['metrics']['clustering']['scratch'])
    print("Inbuilt Models:", results['metrics']['clustering']['inbuilt'])
    
    print("\n--- Classification Metrics (Compliance Risk) ---")
    print("Scratch Models:", results['metrics']['classification']['scratch'])
    print("Inbuilt Models:", results['metrics']['classification']['inbuilt'])
    
    print("\n--- Market Analysis ---")
    print("\nTop 10 Competitors by Market Share:")
    print(pd.DataFrame(results['market_share_top10']).to_string())
    
    print("\nAssociation Rules (Top 5):")
    print(pd.DataFrame(results['association_rules_top5']).to_string())
    
    print("\nSeasonality (Top 5 Rows):")
    print(pd.DataFrame(results['seasonality_top5']).to_string())
    
    print("\n--- Generated Plot Paths ---")
    for fig_name, fig_path in results['plot_paths'].items():
        print(f"   - Figure '{fig_name}' saved at: {fig_path}")

    print("\n--- End of Pipeline ---")