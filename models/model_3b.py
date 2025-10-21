import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

try:
    clients = pd.read_csv("client_contracts.csv")
    bids = pd.read_csv("bids_history.csv")
except FileNotFoundError:
    print("Error: Required CSV files not found.")
    exit()

clients.columns = [c.strip().replace(" ", "_") for c in clients.columns]
bids.columns = [c.strip().replace(" ", "_") for c in bids.columns]

for col in ["Contract_Value_INR"]:
    clients[col] = pd.to_numeric(clients[col], errors='coerce')
for col in ["Bid_Amount_INR", "Competitor_Bid_INR", "Project_Size_sq_ft"]:
    bids[col] = pd.to_numeric(bids[col], errors='coerce')

data = clients.merge(bids, on="Project_ID", how="left")
data["Profitability"] = (data["Contract_Value_INR"] - data["Bid_Amount_INR"]) / (data["Bid_Amount_INR"] + 1e-8)
data["Client_Type"] = data["Client_Type"].fillna("Unknown").astype(str)
client_types_present = data["Client_Type"].unique()

for t in client_types_present:
    col_name = f"ClientType_{t.replace(' ', '_')}"
    data[col_name] = (data["Client_Type"] == t).astype(int)

feature_cols = ["Profitability", "Project_Size_sq_ft", "Contract_Value_INR"] + \
               [f"ClientType_{t.replace(' ', '_')}" for t in client_types_present]

data[feature_cols] = data[feature_cols].fillna(data[feature_cols].median())
X = data[feature_cols].values


class RobustScaler:
    def fit_transform(self, X):
        X = np.array(X, dtype=float)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        median = np.median(X, axis=0)
        iqr = q3 - q1 + 1e-8
        X_scaled = (X - median) / iqr
        return X_scaled

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)


class ManualKMeans:
    def __init__(self, n_clusters=3, n_iter=100):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.centroids = None
        self.labels = None

    def fit(self, X):
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]

        for _ in range(self.n_iter):
            labels = np.array([np.argmin([np.linalg.norm(x - c) for c in self.centroids]) for x in X])
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = cluster_points.mean(axis=0)
                else:
                    new_centroids[k] = self.centroids[k]

            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids

        self.labels = labels

def silhouette_score_manual(X, labels):
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return -1
    score = 0
    for i in range(n_samples):
        same_cluster = X[labels == labels[i]]
        a = np.mean(np.linalg.norm(same_cluster - X[i], axis=1)) if len(same_cluster) > 1 else 0
        b = np.min([np.mean(np.linalg.norm(X[labels == l] - X[i], axis=1)) 
                    for l in unique_labels if l != labels[i]])
        if max(a, b) > 0:
            score += (b - a) / max(a, b)
    return score / n_samples


best_sil = -1
best_k = 2
best_kmeans = None

for k in range(2, 6):
    km = ManualKMeans(n_clusters=k)
    km.fit(X_scaled)
    sil = silhouette_score_manual(X_scaled, km.labels)
    if sil > best_sil:
        best_sil = sil
        best_k = k
        best_kmeans = km

data["Cluster"] = best_kmeans.labels

cluster_summary = data.groupby("Cluster")[["Profitability", "Project_Size_sq_ft", "Contract_Value_INR"]].mean()
for t in client_types_present:
    col = f"ClientType_{t.replace(' ', '_')}"
    if col in data.columns:
        cluster_summary[col] = data.groupby("Cluster")[col].sum()
    else:
        cluster_summary[col] = 0

print("=== Cluster Summary ===")
print(cluster_summary)
print(f"\nBest manual KMeans k={best_k}, silhouette score={best_sil:.4f}")

os.makedirs("artifacts", exist_ok=True)

model_3b = {
    "scaler": scaler,
    "best_kmeans": best_kmeans,
    "best_k": best_k,
    "silhouette_score": best_sil
}

results_3b = {
    "data": data,
    "cluster_summary": cluster_summary
}

with open("artifacts/3b_model.pkl", "wb") as f:
    pickle.dump(model_3b, f)

with open("artifacts/3b_results.pkl", "wb") as f:
    pickle.dump(results_3b, f)

print("\n[Model 3B] Model saved as 'artifacts/3b_model.pkl'")
print("[Model 3B] Results saved as 'artifacts/3b_results.pkl'")

os.makedirs("artifacts", exist_ok=True)
ARTIFACTS_DIR = os.path.abspath("artifacts")

plot_paths = {}

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=data, x="Project_Size_sq_ft", y="Profitability",
    hue="Cluster", palette="Set2", s=100, alpha=0.7
)
plt.title("Clusters: Profitability vs Project Size")
plt.xlabel("Project Size (sq ft)")
plt.ylabel("Profitability")
plt.legend(title="Cluster")
plt.grid(True, linestyle='--', alpha=0.5)
scatter_path = os.path.join(ARTIFACTS_DIR, "3b_cluster_scatter.png")
plt.savefig(scatter_path)
plt.close()

plt.figure(figsize=(8,6))
sns.boxplot(x="Cluster", y="Profitability", data=data, palette="Set3")
plt.title("Profitability Distribution per Cluster")
plt.grid(True, linestyle='--', alpha=0.5)
boxplot_path = os.path.join(ARTIFACTS_DIR, "3b_profitability_boxplot.png")
plt.savefig(boxplot_path)
plt.close()

plt.figure(figsize=(6,5))
sns.countplot(x="Cluster", data=data, palette="pastel")
plt.title("Number of Projects per Cluster")
plt.ylabel("Count")
plt.xlabel("Cluster")
plt.grid(axis='y', linestyle='--', alpha=0.5)
countplot_path = os.path.join(ARTIFACTS_DIR, "3b_cluster_count.png")
plt.savefig(countplot_path)
plt.close()

plt.figure(figsize=(8,6))
corr = data[feature_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
heatmap_path = os.path.join(ARTIFACTS_DIR, "3b_feature_heatmap.png")
plt.savefig(heatmap_path)
plt.close()

from collections import defaultdict

cluster_sil = defaultdict(list)
for i in range(len(data)):
    cluster_sil[data["Cluster"].iloc[i]].append(data["Profitability"].iloc[i]) 
avg_sil = {k: np.mean(v) for k,v in cluster_sil.items()}

plt.figure(figsize=(6,5))
sns.barplot(x=list(avg_sil.keys()), y=list(avg_sil.values()), palette="Set2")
plt.title("Average Profitability per Cluster (proxy silhouette)")
plt.ylabel("Average Profitability")
plt.xlabel("Cluster")
sil_path = os.path.join(ARTIFACTS_DIR, "3b_avg_profitability_per_cluster.png")
plt.savefig(sil_path)
plt.close()
plot_paths["avg_profitability_per_cluster"] = sil_path
