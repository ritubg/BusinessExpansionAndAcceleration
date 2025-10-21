import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


rev = pd.read_csv("financials_revenue.csv")
cash = pd.read_csv("financials_cashflow.csv")

rev.loc[rev['Year'] == 2525, 'Year'] = 2025


X = rev[['Operating_Cost_INR','Net_Profit_INR','Project_Pipeline_Size_INR']].values
y = rev['Revenue_INR'].values

X = np.hstack([np.ones((X.shape[0],1)), X])


train_size = int(len(y)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


def linear_regression(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def predict(X, theta):
    return X @ theta

def evaluate(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred)/y_true))*100
    return mae, rmse, mape

def poly_features(X):
    Xn = X[:,1:]
    poly = [Xn, Xn**2]
    out = np.hstack(poly)
    return np.hstack([np.ones((X.shape[0],1)), out])

def bagging_regressor(X, y, X_test, n_estimators=10, sample_ratio=0.7):
    preds = []
    n = len(y)
    for i in range(n_estimators):
        idx = np.random.choice(n, int(sample_ratio*n), replace=True)
        Xb, yb = X[idx], y[idx]
        theta_b = linear_regression(Xb, yb)
        preds.append(predict(X_test, theta_b))
    return np.mean(preds, axis=0)


theta_lr = linear_regression(X_train, y_train)
pred_lr = predict(X_test, theta_lr)
mae_lr, rmse_lr, mape_lr = evaluate(y_test, pred_lr)

X_poly_train = poly_features(X_train)
X_poly_test = poly_features(X_test)
theta_poly = linear_regression(X_poly_train, y_train)
pred_poly = predict(X_poly_test, theta_poly)
mae_poly, rmse_poly, mape_poly = evaluate(y_test, pred_poly)

pred_bag = bagging_regressor(X_train, y_train, X_test, n_estimators=20)
mae_bag, rmse_bag, mape_bag = evaluate(y_test, pred_bag)


print("Model Performance:")
print(f"Linear Regression -> MAE:{mae_lr:.2f}, RMSE:{rmse_lr:.2f}, MAPE:{mape_lr:.2f}%")
print(f"Polynomial (Quad) -> MAE:{mae_poly:.2f}, RMSE:{rmse_poly:.2f}, MAPE:{mape_poly:.2f}%")
print(f"Ensemble Bagging  -> MAE:{mae_bag:.2f}, RMSE:{rmse_bag:.2f}, MAPE:{mape_bag:.2f}%")

rmse_scores = {"Linear": rmse_lr, "Polynomial": rmse_poly, "Ensemble": rmse_bag}
best_model = min(rmse_scores, key=rmse_scores.get)
print(f"\nBest model chosen: {best_model}")


recent_rev = rev.tail(6)

opcost_growth = recent_rev['Operating_Cost_INR'].pct_change().mean()
profit_growth = recent_rev['Net_Profit_INR'].pct_change().mean()
pipeline_growth = recent_rev['Project_Pipeline_Size_INR'].pct_change().mean()

n_future = 6 
last_opcost = rev['Operating_Cost_INR'].iloc[-1]
last_profit = rev['Net_Profit_INR'].iloc[-1]
last_pipeline = rev['Project_Pipeline_Size_INR'].iloc[-1]

future_opcost = []
future_profit = []
future_pipeline = []

for i in range(n_future):
    last_opcost *= (1 + opcost_growth)
    last_profit *= (1 + profit_growth)
    last_pipeline *= (1 + pipeline_growth)
    future_opcost.append(last_opcost)
    future_profit.append(last_profit)
    future_pipeline.append(last_pipeline)

future_opcost = np.array(future_opcost)
future_profit = np.array(future_profit)
future_pipeline = np.array(future_pipeline)

future_X = np.vstack([future_opcost, future_profit, future_pipeline]).T
future_X = np.hstack([np.ones((future_X.shape[0],1)), future_X])

if best_model == "Linear":
    future_pred = predict(future_X, theta_lr)
elif best_model == "Polynomial":
    future_pred = predict(poly_features(future_X), theta_poly)
else:
    future_pred = bagging_regressor(X_train, y_train, future_X, n_estimators=20)

print("\nForecasted Revenue (INR) for next 6 months:")
for i, rev_val in enumerate(future_pred, 1):
    print(f"Month {i}: {rev_val:,.0f}")


cash['Date'] = pd.to_datetime(cash['Date'], dayfirst=True)
recent_cash = cash.tail(6)
avg_payables = recent_cash['Payables_INR'].mean()
latest_balance = cash['Bank_Balance_INR'].iloc[-1]

liquidity_forecast = [latest_balance]
for rev_month in future_pred:
    new_balance = liquidity_forecast[-1] + rev_month - avg_payables
    liquidity_forecast.append(new_balance)
liquidity_forecast = np.array(liquidity_forecast[1:])

future_dates = pd.date_range(start=cash['Date'].iloc[-1]+pd.offsets.MonthBegin(1), periods=n_future, freq="MS")



plt.figure(figsize=(12,6))
plt.plot(y, "o-", label="Historical Revenue")
plt.plot(range(train_size,len(y)), pred_lr, "r--", label="Linear Reg Test")
plt.plot(range(train_size,len(y)), pred_poly, "g--", label="Poly Reg Test")
plt.plot(range(train_size,len(y)), pred_bag, "c--", label="Bagging Test")
plt.plot(range(len(y), len(y)+len(future_pred)), future_pred, "b-o", label=f"Forecast ({best_model})")
plt.legend()
plt.title("Revenue Prediction with Regression Models")
plt.xlabel("Time Index")
plt.ylabel("Revenue (INR)")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))
plt.plot(cash['Date'], cash['Bank_Balance_INR'], "o-", label="Historical Bank Balance")
plt.plot(future_dates, liquidity_forecast, "r-o", label="Forecasted Liquidity")
plt.legend()
plt.title("Liquidity Forecast Based on Revenue Predictions")
plt.xlabel("Date")
plt.ylabel("Bank Balance (INR)")
plt.grid(True)
plt.show()


growth = ((future_pred[-1]-future_pred[0])/future_pred[0])*100
print("\n--- Insights ---")
print(f"Best model selected: {best_model}")
print(f"Projected Revenue Growth (next {n_future} months): {growth:.2f}%")
print(f"Projected Liquidity Range: {liquidity_forecast.min():,.0f} - {liquidity_forecast.max():,.0f} INR")

import pickle
import os

RESULTS_PATH = "artifacts/4a_results.pkl"

results_4a = {
    "historical_revenue": y.tolist(),
    "pred_lr": pred_lr.tolist(),
    "pred_poly": pred_poly.tolist(),
    "pred_bag": pred_bag.tolist(),
    "future_pred": future_pred.tolist(),
    "future_dates": future_dates.strftime("%Y-%m-%d").tolist(),
    "liquidity_forecast": liquidity_forecast.tolist(),
    "best_model": best_model,
    "metrics": {
        "Linear": {"MAE": mae_lr, "RMSE": rmse_lr, "MAPE": mape_lr},
        "Polynomial": {"MAE": mae_poly, "RMSE": rmse_poly, "MAPE": mape_poly},
        "Bagging": {"MAE": mae_bag, "RMSE": rmse_bag, "MAPE": mape_bag}
    }
}

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, "wb") as f:
    pickle.dump(results_4a, f)

print(f"\nResults saved as '{RESULTS_PATH}'")

artifacts = "artifacts"
os.makedirs(artifacts, exist_ok=True)


plt.figure(figsize=(12,6))
plt.plot(y, "o-", label="Historical Revenue")
plt.plot(range(train_size,len(y)), pred_lr, "r--", label="Linear Reg Test")
plt.plot(range(train_size,len(y)), pred_poly, "g--", label="Poly Reg Test")
plt.plot(range(train_size,len(y)), pred_bag, "c--", label="Bagging Test")
plt.plot(range(len(y), len(y)+len(future_pred)), future_pred, "b-o", label=f"Forecast ({best_model})")
plt.legend()
plt.title("Revenue Prediction with Regression Models")
plt.xlabel("Time Index")
plt.ylabel("Revenue (INR)")
plt.grid(True)
revenue_plot_path = os.path.join(artifacts, "4a_revenue_forecast.png")
plt.savefig(revenue_plot_path)
plt.close() 

plt.figure(figsize=(12,6))
plt.plot(cash['Date'], cash['Bank_Balance_INR'], "o-", label="Historical Bank Balance")
plt.plot(future_dates, liquidity_forecast, "r-o", label="Forecasted Liquidity")
plt.legend()
plt.title("Liquidity Forecast Based on Revenue Predictions")
plt.xlabel("Date")
plt.ylabel("Bank Balance (INR)")
plt.grid(True)
liquidity_plot_path = os.path.join(artifacts, "4a_liquidity_forecast.png")
plt.savefig(liquidity_plot_path)
plt.close()
