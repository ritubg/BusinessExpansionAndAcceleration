import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

class Project2AModel:
    def __init__(self, csv_path="projects_history.csv"):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.model = None
        self.results = None

    def train_models(self, model_path="artifacts/2a_model.pkl"):
        planned_cost = self.df['Planned_Cost_INR'].values
        actual_cost = self.df['Actual_Cost_INR'].values
        X_cost = np.column_stack((np.ones(len(planned_cost)), planned_cost))
        beta_cost = np.linalg.inv(X_cost.T @ X_cost) @ (X_cost.T @ actual_cost)

        delay_flag = (self.df['Actual_Duration_Months'] > self.df['Planned_Duration_Months']).astype(int).values
        X_delay = np.column_stack((np.ones(len(self.df)), self.df['Planned_Duration_Months'].values, planned_cost))
        beta_delay = np.zeros(X_delay.shape[1])
        lr, epochs = 1e-5, 1000
        for _ in range(epochs):
            z = X_delay @ beta_delay
            pred = 1 / (1 + np.exp(-z))
            grad = (X_delay.T @ (pred - delay_flag)) / len(X_delay)
            beta_delay -= lr * grad

        self.model = {"beta_cost": beta_cost, "beta_delay": beta_delay}

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"[2A] Model saved at: {os.path.abspath(model_path)}")
        return self.model

    def generate_plots(self, Ypred_cost, artifacts_dir="artifacts"):
        """Generate comprehensive visualization plots"""
        os.makedirs(artifacts_dir, exist_ok=True)
        sns.set_style("whitegrid")
        plot_paths = {}

        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['Actual_Cost_INR'], Ypred_cost, alpha=0.6, s=50, color='steelblue')
        plt.plot([self.df['Actual_Cost_INR'].min(), self.df['Actual_Cost_INR'].max()],
                 [self.df['Actual_Cost_INR'].min(), self.df['Actual_Cost_INR'].max()],
                 'r--', linewidth=2, label="Perfect Prediction")
        plt.xlabel("Actual Cost (INR)", fontsize=12)
        plt.ylabel("Predicted Cost (INR)", fontsize=12)
        plt.title("Cost Prediction: Actual vs Predicted", fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plot_paths['2a_predicted_vs_actual'] = os.path.abspath(os.path.join(artifacts_dir, "2a_predicted_vs_actual.png"))
        plt.savefig(plot_paths['2a_predicted_vs_actual'], dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        overrun_pct = self.df['PredictedOverRun_percent']
        plt.hist(overrun_pct, bins=30, color='coral', edgecolor='black', alpha=0.7)
        plt.axvline(overrun_pct.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {overrun_pct.mean():.2f}%')
        plt.axvline(overrun_pct.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {overrun_pct.median():.2f}%')
        plt.xlabel("Predicted Cost Overrun (%)", fontsize=12)
        plt.ylabel("Number of Projects", fontsize=12)
        plt.title("Distribution of Predicted Cost Overruns", fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plot_paths['2a_overrun_distribution'] = os.path.abspath(os.path.join(artifacts_dir, "2a_overrun_distribution.png"))
        plt.savefig(plot_paths['2a_overrun_distribution'], dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(self.df['PredictedDelay_prob'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(self.df['PredictedDelay_prob'].mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {self.df["PredictedDelay_prob"].mean():.3f}')
        plt.xlabel("Delay Probability", fontsize=12)
        plt.ylabel("Number of Projects", fontsize=12)
        plt.title("Distribution of Project Delay Probabilities", fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plot_paths['2a_delay_distribution'] =os.path.abspath(os.path.join(artifacts_dir, "2a_delay_distribution.png"))
        plt.savefig(plot_paths['2a_delay_distribution'], dpi=300)
        plt.close()

        plt.figure(figsize=(12, 8))
        top_20 = self.df.nlargest(20, 'Combined_risk')[['Project_ID', 'CostOverrun_risk', 'PredictedDelay_prob']]
        heatmap_data = top_20.set_index('Project_ID')[['CostOverrun_risk', 'PredictedDelay_prob']].T
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Risk Score'})
        plt.title("Risk Heatmap: Top 20 High-Risk Projects", fontsize=14, fontweight='bold')
        plt.xlabel("Project ID", fontsize=12)
        plt.ylabel("Risk Type", fontsize=12)
        plt.tight_layout()
        plot_paths['2a_risk_heatmap'] = os.path.abspath(os.path.join(artifacts_dir, "2a_risk_heatmap.png"))
        plt.savefig(plot_paths['2a_risk_heatmap'], dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(self.df['Combined_risk'], bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
        plt.axvline(self.df['Combined_risk'].mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {self.df["Combined_risk"].mean():.3f}')
        plt.xlabel("Combined Risk Score", fontsize=12)
        plt.ylabel("Number of Projects", fontsize=12)
        plt.title("Distribution of Combined Risk Scores", fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plot_paths['2a_combined_risk_distribution'] = os.path.abspath(os.path.join(artifacts_dir, "2a_combined_risk_distribution.png"))
        plt.savefig(plot_paths['2a_combined_risk_distribution'], dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['Planned_Duration_Months'], self.df['Actual_Duration_Months'], 
                   c=self.df['PredictedDelay_prob'], cmap='RdYlGn_r', s=50, alpha=0.6)
        plt.plot([self.df['Planned_Duration_Months'].min(), self.df['Planned_Duration_Months'].max()],
                 [self.df['Planned_Duration_Months'].min(), self.df['Planned_Duration_Months'].max()],
                 'r--', linewidth=2, label="On Schedule")
        plt.colorbar(label='Delay Probability')
        plt.xlabel("Planned Duration (Months)", fontsize=12)
        plt.ylabel("Actual Duration (Months)", fontsize=12)
        plt.title("Project Duration: Planned vs Actual", fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plot_paths['2a_duration_scatter'] =os.path.abspath(os.path.join(artifacts_dir, "2a_duration_scatter.png"))
        plt.savefig(plot_paths['2a_duration_scatter'], dpi=300)
        plt.close()

        plt.figure(figsize=(12, 6))
        top_10_risky = self.df.nlargest(10, 'Combined_risk')[['Project_ID', 'Combined_risk']]
        plt.barh(top_10_risky['Project_ID'], top_10_risky['Combined_risk'], color='crimson', alpha=0.7)
        plt.xlabel("Combined Risk Score", fontsize=12)
        plt.ylabel("Project ID", fontsize=12)
        plt.title("Top 10 High-Risk Projects", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_paths['2a_top_risky_bar'] =os.path.abspath(os.path.join(artifacts_dir, "2a_top_risky_bar.png"))
        plt.savefig(plot_paths['2a_top_risky_bar'], dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        residuals = self.df['Actual_Cost_INR'] - Ypred_cost
        plt.scatter(Ypred_cost, residuals, alpha=0.6, s=50, color='teal')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel("Predicted Cost (INR)", fontsize=12)
        plt.ylabel("Residuals (Actual - Predicted)", fontsize=12)
        plt.title("Residual Plot: Cost Prediction Errors", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_paths['2a_residual_plot'] = os.path.abspath(os.path.join(artifacts_dir, "2a_residual_plot.png"))
        plt.savefig(plot_paths['2a_residual_plot'], dpi=300)
        plt.close()

        return plot_paths

    def run_predictions(self, model_path="artifacts/2a_model.pkl",
                        savepath="artifacts/2a_results.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        beta_cost = self.model["beta_cost"]
        beta_delay = self.model["beta_delay"]

        X_cost = np.column_stack((np.ones(len(self.df)), self.df['Planned_Cost_INR'].values))
        Ypred_cost = X_cost @ beta_cost
        self.df['Predicted_Actual'] = Ypred_cost
        self.df['PredictedOverRun_percent'] = 100 * ((Ypred_cost - self.df['Planned_Cost_INR']) / self.df['Planned_Cost_INR'])
        
        rmse = np.sqrt(np.mean((self.df['Actual_Cost_INR'] - Ypred_cost) ** 2))
        
        overall_avg = self.df['PredictedOverRun_percent'].mean()
        top_risky = self.df[['Project_ID', 'PredictedOverRun_percent']].sort_values(
            by='PredictedOverRun_percent', ascending=False).head(5)
        safe_projects = self.df[self.df['PredictedOverRun_percent'] <= 0][['Project_ID', 'PredictedOverRun_percent']]

        X_delay = np.column_stack((np.ones(len(self.df)), self.df['Planned_Duration_Months'].values, self.df['Planned_Cost_INR'].values))
        prob_delay = 1 / (1 + np.exp(-(X_delay @ beta_delay)))
        self.df['PredictedDelay_prob'] = prob_delay
        top_delay = self.df[['Project_ID', 'PredictedDelay_prob']].sort_values(
            by='PredictedDelay_prob', ascending=False).head(5)

        self.df['CostOverrun_risk'] = self.df['PredictedOverRun_percent'] / self.df['PredictedOverRun_percent'].max()
        self.df['Combined_risk'] = (self.df['CostOverrun_risk'].fillna(0) + self.df['PredictedDelay_prob']) / 2
        at_risk = self.df[['Project_ID', 'CostOverrun_risk', 'PredictedDelay_prob', 'Combined_risk']].sort_values(
            by='Combined_risk', ascending=False).head(10)

        artifacts_dir = os.path.dirname(savepath) if os.path.dirname(savepath) else "artifacts"
        plot_paths = self.generate_plots(Ypred_cost, artifacts_dir)

        insights = {
            "total_projects": len(self.df),
            "high_risk_projects": len(self.df[self.df['Combined_risk'] > 0.5]),
            "low_risk_projects": len(self.df[self.df['Combined_risk'] <= 0.3]),
            "avg_delay_probability": float(self.df['PredictedDelay_prob'].mean()),
            "max_overrun_percent": float(self.df['PredictedOverRun_percent'].max()),
            "min_overrun_percent": float(self.df['PredictedOverRun_percent'].min()),
            "projects_on_budget": len(self.df[self.df['PredictedOverRun_percent'] <= 0])
        }

        results_dict = {
            "rmse": rmse,
            "overall_avg": overall_avg,
            "top_risky": top_risky,
            "safe_projects": safe_projects,
            "top_delay": top_delay,
            "at_risk": at_risk,
            "plot_paths": plot_paths,
            "insights": insights
        }

        for key in ["top_risky", "safe_projects", "top_delay", "at_risk"]:
            if key in results_dict:
                df = results_dict[key]
                df.replace([np.inf, -np.inf], 0, inplace=True)
                df.fillna(0, inplace=True)
                results_dict[key] = df

        self.results = results_dict

        with open(savepath, "wb") as f:
            pickle.dump(self.results, f)
        print(f"[2A] Results saved at: {os.path.abspath(savepath)}")
        print(f"[2A] Generated {len(plot_paths)} plots in {artifacts_dir}/")
        return self.results

def execute_2a():
    model = Project2AModel()
    model.train_models()
    results = model.run_predictions()
    return results

if __name__ == "__main__":
    execute_2a()