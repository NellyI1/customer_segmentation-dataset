# amazon_price_optimization_model.py

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Step 2: Load the dataset
data = pd.read_csv("/Users/ifeomaigbokwe/Desktop/NEXFORD MSC/BAN 6800/customer_segmentation-dataset/cleaned_amazon_15000.csv")  # Replace with your actual training dataset file path
print("Sample Data:")
print(data.head())

# Step 3: Preprocessing
# Drop unnecessary columns
X = data.drop(['uid', 'asin', 'title', 'price'], axis=1)  # Independent variables
y = data['price']  # Target variable

# Optional: Check for nulls
print("\nMissing values:")
print(data.isnull().sum())

# Step 4: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the XGBoost model
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Results:")
print("MAE:", mae)
print("R2 Score:", r2)

# Step 7: Save the trained model
joblib.dump(model, "xgb_model.joblib")

# Step 8: Scatter Plot - Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="teal")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Scatter Plot: Actual vs Predicted Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Identity line
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.show()

# Step 9: Feature Importance Bar Chart
importances = model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# Step 10: Export Test Results for Reference
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
results_df.to_csv("model_predictions.csv", index=False)

print("\nModel training and evaluation complete. Outputs saved:")
print("- Trained model: xgb_model.joblib")
print("- Scatter plot: scatter_plot.png")
print("- Feature importance chart: feature_importance.png")
print("- Predictions: model_predictions.csv")


