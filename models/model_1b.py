import os
import warnings
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from typing import Dict, Any, Tuple

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

class LaborFeasibilityPipeline:
    
    def __init__(self, raw_dir: str, output_dir: str, zone_dir: str):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.zone_dir = zone_dir
        
        try:
            (
                self.clustered_df, 
                self.labor_df, 
                self.market_df, 
                self.zones_gdf
            ) = self._load_data(raw_dir, output_dir, zone_dir)
            
            print(f"[1B] Loaded 1a cluster data for {len(self.clustered_df)} zones.")
            print(f"[1B] Loaded {len(self.labor_df)} labor records and {len(self.market_df)} market records.")

        except FileNotFoundError as e:
            print(f"[1B] ERROR: Could not load input file. Make sure '1a_clustered_output.pkl' exists in {output_dir}")
            print(f"[1B] Full error: {e}")
            raise
            
        self.processed_df = None
        self.results = None

    def _load_data(self, raw_dir: str, output_dir: str, zone_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
        clustered_df_path = os.path.join(output_dir, "1a_clustered_output.pkl")
        clustered_df = pd.read_pickle(clustered_df_path)
        
        labor_df = pd.read_csv(os.path.join(raw_dir, "labor_market.csv"))
        market_df = pd.read_csv(os.path.join(raw_dir, "market_demand.csv"))
        zones_gdf = gpd.read_file(zone_dir)
        
        return clustered_df, labor_df, market_df, zones_gdf

    def _calculate_feasibility(
        self, 
        clustered_df: pd.DataFrame, 
        labor_df: pd.DataFrame, 
        market_df: pd.DataFrame
    ) -> pd.DataFrame:
        market_df_copy = market_df.copy()
        labor_df_copy = labor_df.copy()
        
        market_df_copy["Ward_id"] = market_df_copy["Ward_id"].astype(int)
        labor_df_copy = labor_df_copy.merge(market_df_copy[["Ward_id", "Zone_Name"]], on="Ward_id", how="left")
        
        zone_labor_agg = labor_df_copy.groupby("Zone_Name").agg({
            "Skilled_Labor_Availability": "sum",
            "Unskilled_Labor_Availability": "sum",
            "Avg_Wage_Skilled": "mean",
            "Avg_Wage_Unskilled": "mean"
        }).reset_index()

        combined_df = clustered_df.merge(zone_labor_agg, on="Zone_Name", how="left")
        
        combined_df["Feasibility_Score"] = (
            (combined_df["Skilled_Labor_Availability"] + combined_df["Unskilled_Labor_Availability"]) /
            (combined_df["Avg_Wage_Skilled"] + combined_df["Avg_Wage_Unskilled"])
        )
        
        combined_df["Feasibility_Score"].fillna(0, inplace=True)
        combined_df["Feasibility_Rank"] = combined_df["Feasibility_Score"].rank(ascending=False, method='dense').astype(int)
        
        return combined_df

    def _generate_plots(self, processed_df: pd.DataFrame, zones_gdf: gpd.GeoDataFrame, artifacts_dir: str) -> Dict[str, str]:
        os.makedirs(artifacts_dir, exist_ok=True)
        plot_paths = {}
        sns.set_style("whitegrid")

        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=processed_df,
            x="Demand_Index",
            y="Feasibility_Score",
            hue="Cluster_Label",
            palette={"Low Potential": "red", "Medium Potential": "orange", "High Potential": "green"},
            s=150,
            edgecolor="black",
            ax=ax_scatter
        )
        ax_scatter.set_title("Zone Demand vs Labor Feasibility", fontsize=14)
        ax_scatter.set_xlabel("Demand Index")
        ax_scatter.set_ylabel("Feasibility Score")
        ax_scatter.legend(title="Cluster", fontsize=10)
        ax_scatter.grid(alpha=0.3)
        
        scatter_path = os.path.join(artifacts_dir, "1b_feasibility_scatter.png")
        fig_scatter.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plot_paths["feasibility_scatter_plot"] = os.path.abspath(scatter_path)
        plt.close(fig_scatter)
        
        zones_gdf_copy = zones_gdf.rename(columns={"ZoneName": "Zone_Name"})
        zones_gdf_copy["Zone_Name"] = zones_gdf_copy["Zone_Name"].str.strip().str.upper()
        processed_df_copy = processed_df.copy()
        processed_df_copy["Zone_Name"] = processed_df_copy["Zone_Name"].str.strip().str.upper()
        
        name_map = {"RAJARAJESHWARI": "RR NAGARA"}
        processed_df_copy["Zone_Name"] = processed_df_copy["Zone_Name"].replace(name_map)
        
        merged_gdf = zones_gdf_copy.merge(
            processed_df_copy[["Zone_Name", "Feasibility_Score", "Cluster_Label"]], 
            on="Zone_Name", 
            how="left"
        )
        
        fig_map, ax_map = plt.subplots(1, 1, figsize=(12, 10))
        merged_gdf.plot(
            column="Feasibility_Score",
            cmap="YlGnBu",
            legend=True,
            ax=ax_map,
            edgecolor="black",
            missing_kwds={"color": "lightgrey"}
        )
        
        for _, row in merged_gdf.iterrows():
            if pd.notna(row["Feasibility_Score"]):
                ax_map.annotate(
                    row["Cluster_Label"],
                    xy=(row["geometry"].centroid.x, row["geometry"].centroid.y),
                    ha="center",
                    fontsize=8,
                    color="black",
                    weight='bold'
                )
                
        ax_map.set_title("Bangalore Zones: Market + Labor Feasibility", fontsize=16)
        ax_map.axis("off")
        
        map_path = os.path.join(artifacts_dir, "1b_feasibility_map.png")
        fig_map.savefig(map_path, dpi=300, bbox_inches='tight')
        plot_paths["feasibility_map_plot"] = os.path.abspath(map_path)
        plt.close(fig_map)
        
        print(f"[1B] Generated {len(plot_paths)} plots in {artifacts_dir}/")
        return plot_paths

    def run_analysis(self, save_path: str, data_save_path: str) -> Dict[str, Any]:
        print("[1B] Calculating feasibility scores...")
        self.processed_df = self._calculate_feasibility(
            self.clustered_df, 
            self.labor_df, 
            self.market_df
        )
        
        artifacts_dir = os.path.dirname(save_path)
        
        print("[1B] Generating plots...")
        plot_paths = self._generate_plots(self.processed_df, self.zones_gdf, artifacts_dir)
        
        self.processed_df.to_pickle(data_save_path)
        print(f"[1B] Processed data saved at: {os.path.abspath(data_save_path)}")
        
        insights_cols = ["Zone_Name", "Demand_Index", "Feasibility_Score", "Feasibility_Rank", "Cluster_Label"]
        final_insights = self.processed_df[insights_cols].sort_values(by="Feasibility_Rank")
        
        self.results = {
            "plot_paths": plot_paths,
            "final_zone_feasibility_insights": final_insights.to_dict('records')
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(self.results, f)
        print(f"[1B] Analysis results saved at: {os.path.abspath(save_path)}")
        
        return self.results

def execute_1b():
    BASE_DIR = r"C:\new\BusinessExpansionAndAcceleration"
    
    RAW_DIR = os.path.join(BASE_DIR, "models")
    ZONE_DIR = os.path.join(BASE_DIR, "models", "Zone", "ZONE.shp")
    
    OUTPUT_DIR = os.path.join(BASE_DIR, "models", "artifacts")
    
    RESULTS_PATH = os.path.join(OUTPUT_DIR, "1b_results.pkl")
    DATA_PATH = os.path.join(OUTPUT_DIR, "1b_feasibility_output.pkl")

    print("--- Running Labor Feasibility Pipeline (1b) ---")
    print(f"  Input Dir (CSVs): {RAW_DIR}")
    print(f"  Input Dir (1a data): {OUTPUT_DIR}")
    print(f"  Output Dir (1b data): {OUTPUT_DIR}")
    
    pipeline = LaborFeasibilityPipeline(
        raw_dir=RAW_DIR,
        output_dir=OUTPUT_DIR,
        zone_dir=ZONE_DIR
    )
    
    results = pipeline.run_analysis(
        save_path=RESULTS_PATH,
        data_save_path=DATA_PATH
    )
    
    return results

if __name__ == "__main__":
    results = execute_1b()
    
    print(f"\n[+] Pipeline execution complete.")

    print("\n--- Final Zone Feasibility Insights (Top 5 Rows) ---")
    insights_df = pd.DataFrame(results['final_zone_feasibility_insights'])
    print(insights_df.head().to_string())
    
    print("\n--- Generated Plot Paths ---")
    for fig_name, fig_path in results['plot_paths'].items():
        print(f"  - Figure '{fig_name}' saved at: {fig_path}")

    print("\n--- End of Pipeline ---")