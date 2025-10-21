import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns


class ClimateDecisionModel:
    def __init__(self,
                 projects_path="projects_history.csv",
                 climate_path="climate_data.csv",
                 materials_path="material_prices.csv",
                 financials_path="financials_revenue.csv"):
        self.projects_path = projects_path
        self.climate_path = climate_path
        self.materials_path = materials_path
        self.financials_path = financials_path
        self.results = {}
        self.model = {}

    def train_model(self, save_model_path="artifacts/2cd_model.pkl", save_results_path="artifacts/2cd_results.pkl"):
        artifacts_dir = "artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)

      
        projects = pd.read_csv(self.projects_path)
        climate = pd.read_csv(self.climate_path)
        materials = pd.read_csv(self.materials_path)
        financials = pd.read_csv(self.financials_path)

        projects.columns = [c.strip().replace(" ", "_") for c in projects.columns]
        projects["Planned_Cost_INR"] = projects["Planned_Cost_INR"].astype(float)
        projects["Planned_Duration_Months"] = projects["Planned_Duration_Months"].astype(float)
        proj_climate = projects.merge(climate, left_on="Location_Ward", right_on="Ward_ID", how="left")

        baseline = materials.iloc[0].copy()
        for col in ["Cement_Price_per_50kg_bag", "Steel_Price_per_kg", "Diesel_Price_per_litre", "Electricity_Price_per_unit"]:
            materials[col + "_Inflation"] = (materials[col] - baseline[col]) / baseline[col]

        inflation_latest = {col: materials[col].iloc[-1] for col in materials.columns if "Inflation" in col}


        def stress_test_modelA(row, inflation, w1=0.6, w2=0.4):
            base_cost = row["Planned_Cost_INR"]
            base_dur = row["Planned_Duration_Months"]

            eco_inflation = (inflation["Cement_Price_per_50kg_bag_Inflation"] + inflation["Steel_Price_per_kg_Inflation"]) / 2
            adj_cost = base_cost * (1 + eco_inflation)

            shock = row.get("Weather_Shock", None)
            if shock == "Heatwave":
                adj_dur = base_dur * 1.15
            elif shock == "Flood":
                adj_dur = base_dur * 1.20
            else:
                adj_dur = base_dur

            score = w1 * ((adj_cost - base_cost) / base_cost) + w2 * ((adj_dur - base_dur) / base_dur)
            return adj_cost, adj_dur, score

        def stress_test_modelB(row, inflation):
            base_cost = row["Planned_Cost_INR"]
            base_dur = row["Planned_Duration_Months"]

            eco_penalty = inflation["Steel_Price_per_kg_Inflation"] * 0.5 + inflation["Diesel_Price_per_litre_Inflation"] * 0.5
            adj_cost = base_cost * (1 + eco_penalty)

            shock = row.get("Weather_Shock", None)
            penalty = 0
            if shock == "Heatwave":
                penalty += 0.10
            elif shock == "Flood":
                penalty += 0.15
            adj_dur = base_dur * (1 + penalty)

            score = ((adj_cost - base_cost) / base_cost + (adj_dur - base_dur) / base_dur) / 2
            return adj_cost, adj_dur, score

        proj_climate[["Adj_Cost_A", "Adj_Dur_A", "Stress_Score_A"]] = proj_climate.apply(
            stress_test_modelA, axis=1, result_type="expand", inflation=inflation_latest
        )

        proj_climate[["Adj_Cost_B", "Adj_Dur_B", "Stress_Score_B"]] = proj_climate.apply(
            stress_test_modelB, axis=1, result_type="expand", inflation=inflation_latest
        )

   
        for model in ["A", "B"]:
            scores = proj_climate[f"Stress_Score_{model}"]
            proj_climate[f"Vulnerability_{model}"] = (scores - scores.min()) / (scores.max() - scores.min())

        compare = proj_climate[["Project_ID", "Stress_Score_A", "Stress_Score_B", "Vulnerability_A", "Vulnerability_B"]]
        mean_diff = (compare["Stress_Score_A"] - compare["Stress_Score_B"]).mean()

   
        financials["Profit_Margin"] = financials["Net_Profit_INR"] / financials["Revenue_INR"]
        avg_profit_margin = financials["Profit_Margin"].mean()

        proj_climate["Stress_Score"] = (proj_climate["Stress_Score_A"] + proj_climate["Stress_Score_B"]) / 2
        proj_climate["Vulnerability"] = (proj_climate["Vulnerability_A"] + proj_climate["Vulnerability_B"]) / 2

        def takeover_decision(row, profit_threshold=0.15, vuln_threshold=0.5):
            projected_profit_margin = avg_profit_margin - row["Stress_Score"] * 0.5 - row["Vulnerability"] * 0.3
            if projected_profit_margin >= profit_threshold and row["Vulnerability"] < vuln_threshold:
                return "Accept"
            elif projected_profit_margin < profit_threshold and row["Vulnerability"] >= vuln_threshold:
                return "Reject"
            else:
                return "Negotiate"

        proj_climate["Decision"] = proj_climate.apply(takeover_decision, axis=1)


        plot_paths = {}

        plt.figure(figsize=(10, 6))
        plt.scatter(proj_climate["Stress_Score"],
                    avg_profit_margin - proj_climate["Stress_Score"] * 0.5 - proj_climate["Vulnerability"] * 0.3,
                    c=pd.Categorical(proj_climate["Decision"]).codes, cmap='viridis', s=100, alpha=0.7)
        plt.xlabel("Stress Score")
        plt.ylabel("Projected Profit Margin")
        plt.title("Profitability vs Stress Score (Decision Coloring)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.colorbar(label="Decision Code")
        path = os.path.abspath(os.path.join(artifacts_dir, "2cd_profit_vs_stress.png"))
        plt.savefig(path); plt.close()
        plot_paths["2cd_profit_vs_stress"] = path

        plt.figure(figsize=(9, 5))
        sns.histplot(proj_climate["Stress_Score"], kde=True, bins=15, color="skyblue")
        plt.title("Distribution of Stress Scores Across Projects")
        plt.xlabel("Stress Score"); plt.ylabel("Frequency")
        path = os.path.abspath(os.path.join(artifacts_dir, "2cd_stress_distribution.png"))
        plt.savefig(path); plt.close()
        plot_paths["2cd_stress_distribution"] = path

        plt.figure(figsize=(8, 5))
        sns.countplot(x="Decision", data=proj_climate, palette="viridis")
        plt.title("Project Decision Counts (Accept / Negotiate / Reject)")
        path = os.path.abspath(os.path.join(artifacts_dir, "2cd_decision_counts.png"))
        plt.savefig(path); plt.close()
        plot_paths["2cd_decision_counts"] = path

        plt.figure(figsize=(10, 6))
        sns.heatmap(proj_climate[["Stress_Score_A", "Stress_Score_B", "Vulnerability_A", "Vulnerability_B"]].corr(),
                    annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap of Stress and Vulnerability Metrics")
        path = os.path.abspath(os.path.join(artifacts_dir, "2cd_correlation_heatmap.png"))
        plt.savefig(path); plt.close()
        plot_paths["2cd_correlation_heatmap"] = path

        plt.figure(figsize=(9, 6))
        sns.boxplot(x="Decision", y="Stress_Score", data=proj_climate, palette="Set2")
        plt.title("Stress Score Distribution by Decision Type")
        path = os.path.abspath(os.path.join(artifacts_dir, "2cd_stress_by_decision.png"))
        plt.savefig(path); plt.close()
        plot_paths["2cd_stress_by_decision"] = path

        plt.figure(figsize=(9, 6))
        sns.violinplot(x="Decision", y="Vulnerability", data=proj_climate, palette="mako")
        plt.title("Vulnerability Distribution by Decision")
        path = os.path.abspath(os.path.join(artifacts_dir, "2cd_vulnerability_violin.png"))
        plt.savefig(path); plt.close()
        plot_paths["2cd_vulnerability_violin"] = path

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="Adj_Cost_A", y="Adj_Dur_A", hue="Decision", data=proj_climate, palette="plasma", s=90)
        plt.title("Adjusted Cost vs Adjusted Duration (Model A)")
        path = os.path.abspath(os.path.join(artifacts_dir, "2cd_cost_vs_duration.png"))
        plt.savefig(path); plt.close()
        plot_paths["2cd_cost_vs_duration"] = path

        sns.pairplot(proj_climate[["Stress_Score", "Vulnerability", "Adj_Cost_A", "Adj_Dur_A"]], diag_kind="kde")
        path = os.path.abspath(os.path.join(artifacts_dir, "2cd_pairplot_features.png"))
        plt.savefig(path); plt.close()
        plot_paths["2cd_pairplot_features"] = path


        shock_counts = climate["Weather_Shock"].value_counts()
        top_shock = shock_counts.idxmax()
        decision_counts = proj_climate["Decision"].value_counts().to_dict()

        insights = {
            "Top_Shock": top_shock,
            "Shock_Count": int(shock_counts[top_shock]),
            "Mean_Stress_Diff": float(mean_diff),
            "Avg_Profit_Margin": float(avg_profit_margin),
            "Decision_Counts": decision_counts
        }

        self.model = {"inflation_latest": inflation_latest, "avg_profit_margin": avg_profit_margin}
        self.results = {
            "compare": compare,
            "proj_climate": proj_climate,
            "insights": insights,
            "plot_paths": plot_paths
        }

        with open(save_model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(save_results_path, "wb") as f:
            pickle.dump(self.results, f)

        print(f"[Model_CD] Model saved at: {os.path.abspath(save_model_path)}")
        print(f"[Model_CD] Results saved at: {os.path.abspath(save_results_path)}")

        return self.results


def execute_model_cd():
    model = ClimateDecisionModel()
    results = model.train_model()
    return results


if __name__ == "__main__":
    execute_model_cd()
