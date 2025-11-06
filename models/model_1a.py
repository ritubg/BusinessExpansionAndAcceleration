import os
import warnings
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
import pickle
from typing import List, Dict, Any, Tuple

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

class MarketClusteringPipeline:
    
    def __init__(self, raw_dir: str, zone_dir: str, csv_name: str):
        print(f"[1A] Initializing pipeline and loading data...")
        self.raw_dir = raw_dir
        self.zone_dir = zone_dir
        self.csv_name = csv_name
        self.market_df, self.zones_gdf = self._load_data(raw_dir, zone_dir, csv_name)
        
        self.processed_df = None
        self.X_scaled = None
        self.model = None
        self.results = None
        print(f"[1A] Loaded {len(self.market_df)} market records from {csv_name} and {len(self.zones_gdf)} zone geometries.")

    def _load_data(self, raw_dir: str, zone_dir: str, csv_name: str) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
        market_df = pd.read_csv(os.path.join(raw_dir, csv_name))
        zones_gdf = gpd.read_file(zone_dir)
        return market_df, zones_gdf

    def _preprocess_data(self, market_df: pd.DataFrame) -> pd.DataFrame:
        df = market_df.copy()
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")

        percentage_cols = ["Decadal_Pop_Growth", "Decadal_HH_Growth"]
        for col in percentage_cols:
            df[col] = df[col].astype(str).str.replace("%", "")
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

        zone_df = df.groupby("Zone_Name").agg({
            "Population": "sum",
            "Households": "sum",
            "Population_Density": "mean",
            "Decadal_Pop_Growth": "mean",
            "Decadal_HH_Growth": "mean",
            "Avg_Income": "mean",
            "Land_cost": "mean",
            "Demand_Index": "mean"
        }).reset_index()

        zone_df["Affordability_Ratio"] = zone_df["Avg_Income"] / zone_df["Land_cost"].replace(0, 1)
        zone_df["Growth_Momentum"] = (zone_df["Decadal_Pop_Growth"] + zone_df["Decadal_HH_Growth"]) / 2
        
        return zone_df

    def _train_scratch_model(self, X: np.ndarray, k: int = 3, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        np.random.seed(random_state)
        centroids = X[np.random.choice(len(X), k, replace=False)]
        
        for _ in range(100):
            labels = np.array([np.argmin([np.sum((x - c)**2) for c in centroids]) for x in X])
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
            if np.all(np.abs(new_centroids - centroids) < 1e-4):
                break
            centroids = new_centroids
            
        cluster_means = {i: np.mean([X[j, 0] for j in range(len(X)) if labels[j] == i]) for i in range(k)}
        ranked_indices = sorted(cluster_means, key=cluster_means.get)
        rank_map = {cluster: rank + 1 for rank, cluster in enumerate(ranked_indices)}
        
        cluster_labels_map = {1: "Low Potential", 2: "Medium Potential", 3: "High Potential"}
        cluster_rank = np.array([rank_map[label] for label in labels])
        cluster_label = np.array([cluster_labels_map[rank] for rank in cluster_rank])
        
        inertia = np.sum([np.sum((X[labels == i] - centroids[i])**2) for i in range(k)])
        
        return labels, cluster_rank, cluster_label, centroids, inertia

    def _train_inbuilt_model(self, X: pd.DataFrame, k: int = 3, random_state: int = 42) -> Pipeline:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=k, random_state=random_state, n_init=10))
        ])
        pipeline.fit(X)
        return pipeline

    def _evaluate_model(self, X_scaled: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        if len(np.unique(labels)) > 1:
            metrics = {
                "silhouette_score": float(silhouette_score(X_scaled, labels)),
                "davies_bouldin_score": float(davies_bouldin_score(X_scaled, labels)),
                "calinski_harabasz_score": float(calinski_harabasz_score(X_scaled, labels))
            }
        else:
            metrics = {
                "silhouette_score": 0.0,
                "davies_bouldin_score": 0.0,
                "calinski_harabasz_score": 0.0
            }
        return metrics

    def train_model(self, model_path: str = "outputs/1a_market_cluster.pkl", k: int = 3, random_state: int = 42) -> Pipeline:
        self.processed_df = self._preprocess_data(self.market_df)
        
        features_to_scale = ["Demand_Index", "Affordability_Ratio", "Growth_Momentum"]
        X = self.processed_df[features_to_scale]
        
        self.model = self._train_inbuilt_model(X, k=k, random_state=random_state)
        
        self.processed_df["Cluster"] = self.model.predict(X)
        self.X_scaled = self.model.named_steps['scaler'].transform(X)

        cluster_means = self.processed_df.groupby("Cluster")["Demand_Index"].mean().sort_values()
        cluster_rank_map = {cluster: rank + 1 for rank, cluster in enumerate(cluster_means.index)}
        self.processed_df["Cluster_Rank"] = self.processed_df["Cluster"].map(cluster_rank_map)
        self.processed_df["Cluster_Label"] = self.processed_df["Cluster_Rank"].map({
            1: "Low Potential", 2: "Medium Potential", 3: "High Potential"
        })
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        
        return self.model

    def _generate_plots(self, artifacts_dir: str = "outputs") -> Dict[str, str]:
        os.makedirs(artifacts_dir, exist_ok=True)
        plot_paths = {}
        sns.set_style("whitegrid")

        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=self.processed_df,
            x="Demand_Index",
            y="Affordability_Ratio",
            hue="Cluster_Label",
            palette={"Low Potential": "red", "Medium Potential": "orange", "High Potential": "green"},
            s=150,
            edgecolor="black",
            ax=ax_scatter
        )
        ax_scatter.set_title("Zone Clusters: Demand vs Affordability", fontsize=14)
        ax_scatter.set_xlabel("Demand Index")
        ax_scatter.set_ylabel("Affordability Ratio")
        ax_scatter.legend(title="Cluster", fontsize=10)
        ax_scatter.grid(alpha=0.3)
        
        scatter_path = os.path.join(artifacts_dir, "1a_scatter_plot.png")
        fig_scatter.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plot_paths["scatter_plot"] = os.path.abspath(scatter_path)
        plt.close(fig_scatter)

        zones_gdf_copy = self.zones_gdf.rename(columns={"ZoneName": "Zone_Name"})
        zones_gdf_copy["Zone_Name"] = zones_gdf_copy["Zone_Name"].str.strip().str.upper()
        processed_df_copy = self.processed_df.copy()
        processed_df_copy["Zone_Name"] = processed_df_copy["Zone_Name"].str.strip().str.upper()
        
        name_map = {"RAJARAJESHWARI": "RR NAGARA"}
        processed_df_copy["Zone_Name"] = processed_df_copy["Zone_Name"].replace(name_map)
        
        merged_gdf = zones_gdf_copy.merge(processed_df_copy[["Zone_Name", "Cluster_Label"]], on="Zone_Name", how="left")
        
        fig_map, ax_map = plt.subplots(1, 1, figsize=(10, 8))
        merged_gdf.plot(
            column="Cluster_Label",
            cmap="viridis",
            legend=True,
            ax=ax_map,
            edgecolor="black",
            missing_kwds={"color": "lightgrey", "label": "Missing"}
        )
        ax_map.set_title("Bangalore Zone Clustering (Market Potential)", fontsize=14)
        ax_map.axis("off")

        map_path = os.path.join(artifacts_dir, "1a_map_plot.png")
        fig_map.savefig(map_path, dpi=300, bbox_inches='tight')
        plot_paths["map_plot"] = os.path.abspath(map_path)
        plt.close(fig_map)
        
        return plot_paths

    def run_analysis(self, model_path: str, save_path: str, data_save_path: str):
        
        if self.model is None:
            self.model = joblib.load(model_path)
        
        if self.processed_df is None or "Cluster" not in self.processed_df.columns:
            self.processed_df = self._preprocess_data(self.market_df)
            features_to_scale = ["Demand_Index", "Affordability_Ratio", "Growth_Momentum"]
            X = self.processed_df[features_to_scale]
            self.processed_df["Cluster"] = self.model.predict(X)
            self.X_scaled = self.model.named_steps['scaler'].transform(X)
            
            cluster_means = self.processed_df.groupby("Cluster")["Demand_Index"].mean().sort_values()
            cluster_rank_map = {cluster: rank + 1 for rank, cluster in enumerate(cluster_means.index)}
            self.processed_df["Cluster_Rank"] = self.processed_df["Cluster"].map(cluster_rank_map)
            self.processed_df["Cluster_Label"] = self.processed_df["Cluster_Rank"].map({
                1: "Low Potential", 2: "Medium Potential", 3: "High Potential"
            })
        
        s_labels, s_rank, s_label_text, _, s_inertia = self._train_scratch_model(self.X_scaled, k=3, random_state=42)
        self.processed_df["Cluster_scratch"] = s_labels
        self.processed_df["Cluster_Rank_scratch"] = s_rank
        self.processed_df["Cluster_Label_scratch"] = s_label_text

        metrics_inbuilt = self._evaluate_model(self.X_scaled, self.processed_df["Cluster"].values)
        metrics_inbuilt['inertia'] = float(self.model.named_steps['kmeans'].inertia_)
        
        metrics_scratch = self._evaluate_model(self.X_scaled, self.processed_df["Cluster_scratch"].values)
        metrics_scratch['inertia'] = float(s_inertia)
        
        metrics_dict = {
            "inbuilt_model": metrics_inbuilt,
            "scratch_model": metrics_scratch
        }
        
        artifacts_dir = os.path.dirname(save_path)
        plot_paths = self._generate_plots(artifacts_dir)
        print(f"[1A] Generated {len(plot_paths)} plots in {artifacts_dir}/")

        self.processed_df.to_pickle(data_save_path)
        print(f"[1A] Processed data saved at: {os.path.abspath(data_save_path)}")
        
        top_5_cols = ["Zone_Name", "Demand_Index", "Affordability_Ratio", "Growth_Momentum", "Cluster_Label"]
        
        self.results = {
            "metrics": metrics_dict,
            "plot_paths": plot_paths,
            "final_zone_insights": self.processed_df[top_5_cols].to_dict('records'),
            "scratch_kmeans_results": self.processed_df[["Zone_Name", "Cluster_scratch", "Cluster_Rank_scratch", "Cluster_Label_scratch"]].to_dict('records'),
            "inbuilt_kmeans_results": self.processed_df[["Zone_Name", "Cluster", "Cluster_Rank", "Cluster_Label"]].to_dict('records')
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(self.results, f)        
        return self.results


def execute_1a():
    BASE_DIR = r"C:\new\BusinessExpansionAndAcceleration"
    
    RAW_DIR = os.path.join(BASE_DIR, "models")
    CSV_NAME = "market_demand.csv"
    
    ZONE_DIR = os.path.join(BASE_DIR, "models", "Zone", "ZONE.shp")
    
    OUTPUT_DIR = os.path.join(BASE_DIR, "models", "artifacts")
    
    MODEL_PATH = os.path.join(OUTPUT_DIR, "1a_market_cluster.pkl")
    RESULTS_PATH = os.path.join(OUTPUT_DIR, "1a_results.pkl")
    DATA_PATH = os.path.join(OUTPUT_DIR, "1a_clustered_output.pkl")

    print("--- Running Market Clustering Pipeline (1a) ---")
    print(f"  Input CSV: {os.path.join(RAW_DIR, CSV_NAME)}")
    print(f"  Input Zone: {ZONE_DIR}")
    print(f"  Output Dir: {OUTPUT_DIR}")
    
    pipeline = MarketClusteringPipeline(
        raw_dir=RAW_DIR, 
        zone_dir=ZONE_DIR, 
        csv_name=CSV_NAME
    )
    
    pipeline.train_model(model_path=MODEL_PATH)
    
    results = pipeline.run_analysis(
        model_path=MODEL_PATH,
        save_path=RESULTS_PATH,
        data_save_path=DATA_PATH
    )
    
    return results

if __name__ == "__main__":
    results = execute_1a()