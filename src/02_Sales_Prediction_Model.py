import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
data = pd.read_csv("Advertising_Cleaned.csv")
features = ["TV", "Radio", "Newspaper"]
X, y = data[features], data["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {"Linear Regression": LinearRegression(), "Random Forest Regressor": RandomForestRegressor(n_estimators=400, random_state=42)}
results, fitted, predictions = [], {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    fitted[name], predictions[name] = model, pred
    results.append({"Model": name, "MAE": mean_absolute_error(y_test, pred), "RMSE": mean_squared_error(y_test, pred)**0.5, "R2_Score": r2_score(y_test, pred)})
metrics = pd.DataFrame(results).sort_values("R2_Score", ascending=False)
metrics.to_csv("Model_Evaluation_Metrics.csv", index=False)
best_name = metrics.iloc[0]["Model"]
best_model, best_pred = fitted[best_name], predictions[best_name]
pd.DataFrame({"Actual_Sales": y_test.values, "Predicted_Sales": best_pred}).to_csv("Test_Set_Predictions.csv", index=False)
perm = permutation_importance(best_model, X_test, y_test, n_repeats=30, random_state=42, scoring="r2")
pd.DataFrame({"Feature": features, "Importance_Mean": perm.importances_mean, "Importance_Std": perm.importances_std}).sort_values("Importance_Mean", ascending=False).to_csv("Feature_Importance.csv", index=False)
plt.figure(figsize=(6,6)); plt.scatter(y_test, best_pred, alpha=0.8); lim=[min(y_test.min(),best_pred.min()),max(y_test.max(),best_pred.max())]; plt.plot(lim,lim,linestyle="--"); plt.xlabel("Actual Sales"); plt.ylabel("Predicted Sales"); plt.tight_layout(); plt.savefig("Actual_vs_Predicted.png",dpi=180)
print(metrics.to_string(index=False)); print("Selected model:", best_name)
