import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

os.makedirs("artifacts", exist_ok=True)
artifacts_dir = os.path.abspath("artifacts")


contracts = pd.read_csv("client_contracts.csv")
contracts.columns = [c.strip().replace(" ", "_") for c in contracts.columns]

contracts["Contract_Value_INR"] = contracts["Contract_Value_INR"].astype(float)
contracts["Start_Date"] = pd.to_datetime(contracts["Start_Date"])
contracts["End_Date"] = pd.to_datetime(contracts["End_Date"])
contracts["Contract_Duration_Months"] = ((contracts["End_Date"] - contracts["Start_Date"]).dt.days) / 30

for t in contracts["Contract_Type"].unique():
    contracts[f"Contract_Type_{t.replace(' ','_')}"] = (contracts["Contract_Type"] == t).astype(int)

contracts_model = contracts.drop(columns=["Contract_Type","Start_Date","End_Date","Status","Project_ID","Contract_ID","Client_ID"])

contracts_model["Churn"] = (contracts_model["Contract_Duration_Months"] > contracts_model["Contract_Duration_Months"].mean()).astype(int)

X = contracts_model.drop(columns=["Churn"]).values
y = contracts_model["Churn"].values.reshape(-1,1)

X_min, X_max = X.min(axis=0), X.max(axis=0)
X_scaled = (X - X_min) / (X_max - X_min + 1e-8)


class ManualLogisticRegression:
    def __init__(self, lr=0.1, n_iter=10000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features,1))
        self.bias = 0
        for _ in range(self.n_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_prob(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)


class ManualRandomForest:
    def __init__(self, n_trees=10, max_depth=3, sample_frac=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_frac = sample_frac
        self.trees = []

    class TreeNode:
        def __init__(self, gini, feature=None, threshold=None, left=None, right=None, value=None):
            self.gini = gini
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def gini(self, y):
        p1 = y.mean()
        return 1 - p1**2 - (1-p1)**2

    def split(self, X, y, feature, threshold):
        left_idx = X[:,feature] <= threshold
        right_idx = X[:,feature] > threshold
        return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

    def best_split(self, X, y):
        n_features = X.shape[1]
        best_gini = 1
        best_feat, best_thresh = None, None
        for f in range(n_features):
            thresholds = np.unique(X[:,f])
            for t in thresholds:
                _, y_l, _, y_r = self.split(X, y, f, t)
                if len(y_l)==0 or len(y_r)==0:
                    continue
                gini_split = (len(y_l)*self.gini(y_l) + len(y_r)*self.gini(y_r))/len(y)
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_feat = f
                    best_thresh = t
        return best_feat, best_thresh, best_gini

    def build_tree(self, X, y, depth=0):
        if depth>=self.max_depth or len(np.unique(y))==1:
            return self.TreeNode(gini=self.gini(y), value=y.mean())
        f, t, g = self.best_split(X, y)
        if f is None:
            return self.TreeNode(gini=self.gini(y), value=y.mean())
        X_l, y_l, X_r, y_r = self.split(X, y, f, t)
        left = self.build_tree(X_l, y_l, depth+1)
        right = self.build_tree(X_r, y_r, depth+1)
        return self.TreeNode(gini=g, feature=f, threshold=t, left=left, right=right)

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_trees):
            idx = np.random.choice(n_samples, int(n_samples*self.sample_frac), replace=True)
            X_s, y_s = X[idx], y[idx]
            tree = self.build_tree(X_s, y_s)
            self.trees.append(tree)

    def predict_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_tree(x, node.left)
        else:
            return self.predict_tree(x, node.right)

    def predict_prob(self, X):
        preds = np.array([[self.predict_tree(x, t) for t in self.trees] for x in X])
        return preds.mean(axis=1).reshape(-1,1)

    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)


class ManualGradientBoosting:
    def __init__(self, n_estimators=5, lr=0.1, max_depth=2):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        pred = np.full(y.shape, y.mean())
        for _ in range(self.n_estimators):
            residual = y - pred
            tree = ManualRandomForest(n_trees=1, max_depth=self.max_depth)
            tree.fit(X, residual)
            update = tree.predict_prob(X)
            pred += self.lr * update
            self.trees.append(tree)

    def predict_prob(self, X):
        pred = 0
        for tree in self.trees:
            pred += self.lr * tree.predict_prob(X)
        return 1/(1+np.exp(-pred))

    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)


def accuracy(y_true, y_pred): return (y_true==y_pred).mean()

log_model = ManualLogisticRegression(lr=0.5, n_iter=10000)
log_model.fit(X_scaled, y)
log_acc = accuracy(y, log_model.predict(X_scaled))

rf_model = ManualRandomForest(n_trees=10, max_depth=3)
rf_model.fit(X_scaled, y)
rf_acc = accuracy(y, rf_model.predict(X_scaled))

gb_model = ManualGradientBoosting(n_estimators=5, lr=0.1, max_depth=2)
gb_model.fit(X_scaled, y)
gb_acc = accuracy(y, gb_model.predict(X_scaled))

metrics = {"LogisticRegression": log_acc, "RandomForest": rf_acc, "GradientBoosting": gb_acc}
best_model_name = max(metrics, key=metrics.get)
best_model = {"LogisticRegression": log_model, "RandomForest": rf_model, "GradientBoosting": gb_model}[best_model_name]


contracts_full = pd.read_csv("client_contracts.csv")
contracts_full.columns = [c.strip().replace(" ", "_") for c in contracts_full.columns]
contracts_full["Churn_Prob"] = best_model.predict_prob(X_scaled)
contracts_full["High_Churn_Risk"] = (contracts_full["Churn_Prob"] > 0.5).astype(int)

def churn_action(row):
    if row["High_Churn_Risk"]==1:
        return "Immediate Engagement" if row["Contract_Value_INR"]>500_000_000 else "Monitor & Retain"
    else:
        return "Normal Monitoring"

contracts_full["Action_Recommendation"] = contracts_full.apply(churn_action, axis=1)


MODEL_PATH = os.path.join(artifacts_dir, "3a_model.pkl")
RESULTS_PATH = os.path.join(artifacts_dir, "3a_results.pkl")

with open(MODEL_PATH, "wb") as f: pickle.dump(best_model, f)
with open(RESULTS_PATH, "wb") as f:
    pickle.dump({
        "metrics": metrics,
        "best_model_name": best_model_name,
        "client_report": contracts_full
    }, f)

print(f"[3A Model] Best model saved at: {MODEL_PATH}")
print(f"[3A Results] Client report saved at: {RESULTS_PATH}")


plt.figure(figsize=(6,4))
plt.bar(metrics.keys(), metrics.values(), color=["skyblue","lightgreen","salmon"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(artifacts_dir, "3a_model_accuracy.png"))
plt.close()

plt.figure(figsize=(6,4))
plt.hist(contracts_full["Churn_Prob"], bins=15, color="mediumpurple", edgecolor="black")
plt.title("Distribution of Churn Probabilities")
plt.xlabel("Predicted Churn Probability")
plt.ylabel("Number of Clients")
plt.tight_layout()
plt.savefig(os.path.join(artifacts_dir, "3a_churn_distribution.png"))
plt.close()

plt.figure(figsize=(6,4))
plt.scatter(contracts_full["Contract_Value_INR"], contracts_full["Churn_Prob"], c=contracts_full["High_Churn_Risk"], cmap="coolwarm", alpha=0.7)
plt.title("Contract Value vs Churn Probability")
plt.xlabel("Contract Value (INR)")
plt.ylabel("Churn Probability")
plt.tight_layout()
plt.savefig(os.path.join(artifacts_dir, "3a_value_vs_churn.png"))
plt.close()

plt.figure(figsize=(6,4))
contracts_full["Action_Recommendation"].value_counts().plot(kind="bar", color="teal")
plt.title("Action Recommendation Distribution")
plt.xlabel("Action Type")
plt.ylabel("Number of Clients")
plt.tight_layout()
plt.savefig(os.path.join(artifacts_dir, "3a_action_summary.png"))
plt.close()

plt.figure(figsize=(8,6))
corr = contracts_model.corr()
plt.imshow(corr, cmap='coolwarm', interpolation='none')
plt.colorbar(label="Correlation")
plt.title("Feature Correlation Heatmap")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.savefig(os.path.join(artifacts_dir, "3a_feature_heatmap.png"))
plt.close()

print("All 3A plots saved in artifacts directory successfully.")
