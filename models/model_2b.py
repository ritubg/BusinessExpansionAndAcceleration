import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import mean_squared_error

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.feature_cols = None
        self.scalers = {}

    def one_hot_encode(self, column):
        unique_values = self.df[column].unique()
        for value in unique_values:
            col_name = f"{column}_{value.replace(' ', '_').replace('/', '_')}"
            self.df[col_name] = (self.df[column] == value).astype(int)
        self.df.drop(columns=[column], inplace=True)

    def min_max_scale(self, data, name):
        min_val = data.min()
        max_val = data.max()
        self.scalers[name] = {'min': min_val, 'max': max_val}
        return (data - min_val) / (max_val - min_val)

    def inverse_transform(self, scaled_data, name):
        scaler = self.scalers[name]
        return scaled_data * (scaler['max'] - scaler['min']) + scaler['min']

    def preprocess(self):
        self.one_hot_encode('Project_Type')
        self.feature_cols = ['Planned_Cost_INR', 'Planned_Duration_Months'] + \
                            [col for col in self.df.columns if 'Project_Type_' in col]

        X = self.df[self.feature_cols].values
        y_cost = self.df['Actual_Cost_INR'].values.reshape(-1,1)
        y_timeline = self.df['Actual_Duration_Months'].values.reshape(-1,1)

        X_scaled = self.min_max_scale(X, 'features')
        y_cost_scaled = self.min_max_scale(y_cost, 'cost')
        y_timeline_scaled = self.min_max_scale(y_timeline, 'timeline')

        return X_scaled, y_cost_scaled, y_timeline_scaled


class BaseModel:
    def fit(self, X, y): raise NotImplementedError
    def predict(self, X): raise NotImplementedError

class ManualLinearRegression(BaseModel):
    def __init__(self, lr=0.01, n_iter=10000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
    def __str__(self): return "Manual Linear Regression"

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y = y.flatten()
        for _ in range(self.n_iter):
            y_pred = X @ self.weights + self.bias
            dw = (X.T @ (y_pred - y)) / n_samples
            db = np.mean(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return X @ self.weights + self.bias


class ManualKNNRegression(BaseModel):
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    def __str__(self): return f"Manual KNN Regression (k={self.k})"

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y.flatten()

    def predict(self, X):
        preds = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            k_idx = np.argsort(distances)[:self.k]
            preds.append(np.mean(self.y_train[k_idx]))
        return np.array(preds)


class ModelEvaluator:
    def mean_absolute_error(self, y_true, y_pred):
        return np.mean(np.abs(y_true.flatten() - y_pred.flatten()))

    def compare_models(self, models, X_test, y_cost_test, y_timeline_test, preprocessor):
        results = {}
        for model in models:
            cost_pred_scaled = model.predict(X_test)
            timeline_pred_scaled = model.predict(X_test)

            if isinstance(model, ManualLinearRegression):
                cost_pred = preprocessor.inverse_transform(cost_pred_scaled.reshape(-1,1), 'cost')
                timeline_pred = preprocessor.inverse_transform(timeline_pred_scaled.reshape(-1,1), 'timeline')
            else:
                cost_pred = cost_pred_scaled.reshape(-1,1)
                timeline_pred = timeline_pred_scaled.reshape(-1,1)

            cost_mae = self.mean_absolute_error(y_cost_test, cost_pred)
            timeline_mae = self.mean_absolute_error(y_timeline_test, timeline_pred)
            avg_mae = (cost_mae + timeline_mae) / 2

            results[str(model)] = {
                'cost_mae': cost_mae,
                'timeline_mae': timeline_mae,
                'avg_mae': avg_mae,
                'cost_predictions': cost_pred,
                'timeline_predictions': timeline_pred
            }

        best_model_name = min(results, key=lambda k: results[k]['avg_mae'])
        return best_model_name, results


class InsightsGenerator:
    def __init__(self, df, preprocessor):
        self.df = df
        self.preprocessor = preprocessor

    def project_overrun_analysis(self):
        self.df['Cost_Overrun'] = self.df['Actual_Cost_INR'] - self.df['Planned_Cost_INR']
        self.df['Time_Overrun'] = self.df['Actual_Duration_Months'] - self.df['Planned_Duration_Months']
        avg_cost_overrun = self.df['Cost_Overrun'].mean()
        avg_time_overrun = self.df['Time_Overrun'].mean()
        worst_type = self.df.groupby("Project_Type")['Cost_Overrun'].mean().idxmax()
        return {
            "avg_cost_overrun": avg_cost_overrun,
            "avg_time_overrun": avg_time_overrun,
            "worst_type": worst_type
        }

class Visualizer:
    def plot_scatter(self, y_true, y_pred, title, y_label, save_path):
        plt.figure(figsize=(10,6))
        plt.scatter(y_true, y_pred, color='dodgerblue', s=100, alpha=0.7, label='Predictions')
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        plt.plot([min_val,max_val],[min_val,max_val],'k--', lw=2, label='Perfect Prediction')
        plt.title(f"{title} - Actual vs Predicted", fontsize=16)
        plt.xlabel(f"Actual {y_label}", fontsize=12)
        plt.ylabel(f"Predicted {y_label}", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_histogram(self, data, title, x_label, save_path, bins=10):
        plt.figure(figsize=(10,6))
        plt.hist(data, bins=bins, color='orange', edgecolor='black', alpha=0.7)
        plt.title(title, fontsize=16)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def run_pipeline(csv_path="projects_history.csv", artifacts_dir="artifacts", savepath="artifacts/2b_results.pkl"):
    df = pd.read_csv(csv_path)

    preprocessor = DataPreprocessor(df)
    X_scaled, y_cost_scaled, y_timeline_scaled = preprocessor.preprocess()

    split = int(len(df)*0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_cost_train, y_cost_test = y_cost_scaled[:split], y_cost_scaled[split:]
    y_timeline_train, y_timeline_test = y_timeline_scaled[:split], y_timeline_scaled[split:]

    lr = ManualLinearRegression()
    knn = ManualKNNRegression(k=3)
    lr.fit(X_train, y_cost_train)
    knn.fit(X_train, y_cost_train)

    evaluator = ModelEvaluator()
    best_model_name, results = evaluator.compare_models(
        [lr, knn], X_test,
        preprocessor.inverse_transform(y_cost_test,'cost'),
        preprocessor.inverse_transform(y_timeline_test,'timeline'),
        preprocessor
    )

    insights = InsightsGenerator(df, preprocessor)
    overrun_info = insights.project_overrun_analysis()

    visualizer = Visualizer()
    best_model_results = results[best_model_name]

    plot_paths = {}

    visualizer.plot_scatter(preprocessor.inverse_transform(y_cost_test,'cost'),
                            best_model_results['cost_predictions'],
                            "Cost Prediction", "Cost (INR)",
                            os.path.join(artifacts_dir, "2b_cost_predictions.png"))
    plot_paths['2b_cost_predictions'] = os.path.abspath(os.path.join(artifacts_dir, "2b_cost_predictions.png"))

    visualizer.plot_scatter(preprocessor.inverse_transform(y_timeline_test,'timeline'),
                            best_model_results['timeline_predictions'],
                            "Timeline Prediction", "Duration (Months)",
                            os.path.join(artifacts_dir, "2b_timeline_predictions.png"))
    plot_paths['2b_timeline_predictions'] = os.path.abspath(os.path.join(artifacts_dir, "2b_timeline_predictions.png"))

    visualizer.plot_histogram(df['Actual_Cost_INR'] - df['Planned_Cost_INR'],
                              "Cost Overrun Distribution", "Cost Overrun (INR)",
                              os.path.join(artifacts_dir, "2b_cost_overrun_dist.png"))
    plot_paths['2b_cost_overrun_dist'] = os.path.abspath(os.path.join(artifacts_dir, "2b_cost_overrun_dist.png"))

    visualizer.plot_histogram(df['Actual_Duration_Months'] - df['Planned_Duration_Months'],
                              "Timeline Overrun Distribution", "Timeline Overrun (Months)",
                              os.path.join(artifacts_dir, "2b_timeline_overrun_dist.png"))
    plot_paths['2b_timeline_overrun_dist'] = os.path.abspath(os.path.join(artifacts_dir, "2b_timeline_overrun_dist.png"))

    visualizer.plot_scatter(df['Planned_Cost_INR'], df['Actual_Cost_INR'],
                            "Planned vs Actual Cost", "Cost (INR)",
                            os.path.join(artifacts_dir, "2b_cost_scatter.png"))
    plot_paths['2b_cost_scatter'] = os.path.abspath(os.path.join(artifacts_dir, "2b_cost_scatter.png"))

    visualizer.plot_scatter(df['Planned_Duration_Months'], df['Actual_Duration_Months'],
                            "Planned vs Actual Timeline", "Duration (Months)",
                            os.path.join(artifacts_dir, "2b_timeline_scatter.png"))
    plot_paths['2b_timeline_scatter'] = os.path.abspath(os.path.join(artifacts_dir, "2b_timeline_scatter.png"))

    visualizer.plot_scatter(preprocessor.inverse_transform(y_cost_test,'cost') - best_model_results['cost_predictions'],
                            best_model_results['cost_predictions'],
                            "Cost Residual Plot", "Predicted Cost",
                            os.path.join(artifacts_dir, "2b_residual_plot_cost.png"))
    plot_paths['2b_residual_plot_cost'] = os.path.abspath(os.path.join(artifacts_dir, "2b_residual_plot_cost.png"))

    visualizer.plot_scatter(preprocessor.inverse_transform(y_timeline_test,'timeline') - best_model_results['timeline_predictions'],
                            best_model_results['timeline_predictions'],
                            "Timeline Residual Plot", "Predicted Timeline",
                            os.path.join(artifacts_dir, "2b_residual_plot_timeline.png"))
    plot_paths['2b_residual_plot_timeline'] = os.path.abspath(os.path.join(artifacts_dir, "2b_residual_plot_timeline.png"))

    cost_rmse = np.sqrt(mean_squared_error(
        preprocessor.inverse_transform(y_cost_test,'cost'),
        best_model_results['cost_predictions']
    ))
    timeline_rmse = np.sqrt(mean_squared_error(
        preprocessor.inverse_transform(y_timeline_test,'timeline'),
        best_model_results['timeline_predictions']
    ))

    top_risky_cost = df.sort_values('Actual_Cost_INR', ascending=False).head(5)
    safe_projects_cost = df.sort_values('Actual_Cost_INR', ascending=True).head(5)

    all_results = {
        "cost_rmse": cost_rmse,
        "timeline_rmse": timeline_rmse,
        "top_risky_cost": top_risky_cost,
        "safe_projects_cost": safe_projects_cost,
        "plot_paths": plot_paths,
        "insights": overrun_info
    }

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, "wb") as f:
        pickle.dump(all_results, f)

    print(f"Pipeline complete! Results saved to {savepath}")
    return all_results

if __name__ == "__main__":
    run_pipeline()
