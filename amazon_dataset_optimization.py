# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Load the Amazon Dataset
try:
    amazon_df = pd.read_csv("/Users/ifeomaigbokwe/Desktop/NEXFORD MSC/BAN 6800/customer_segmentation-dataset/random_sample_15000_records.csv")
except Exception as e:
    print("❌ Error loading file:", e)
    exit()

# Step 3: Preview the dataset
print("✅ Amazon Dataset Loaded Successfully")
print("📄 Column Names:", amazon_df.columns.tolist())
print("\n🔍 First Five Rows:")
print(amazon_df.head())

# Step 4: Check for missing values
print("\n🧹 Missing Values in Each Column:")
print(amazon_df.isnull().sum())

# Step 5: Handle Missing Data
num_cols = amazon_df.select_dtypes(include=[np.number]).columns.tolist()
amazon_df[num_cols] = amazon_df[num_cols].fillna(amazon_df[num_cols].median())

cat_cols = amazon_df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    amazon_df[col] = amazon_df[col].fillna(amazon_df[col].mode()[0])

print("\n✅ Missing values handled.")

# Step 6: Feature Engineering
print("\n🛠️ Feature Engineering:")

# Feature 1: Price per review
amazon_df['Price_per_Review'] = amazon_df['price'] / (amazon_df['reviews'] + 1)
print("- Created 'Price_per_Review'")

# Feature 2: High rating (1 if stars >= 4.5)
amazon_df['High_Rating'] = (amazon_df['stars'] >= 4.5).astype(int)
print("- Created 'High_Rating' flag")

# Feature 3: Log transform of price to reduce skew
amazon_df['Log_Price'] = np.log1p(amazon_df['price'])
print("- Created 'Log_Price'")

# Step 7: Encode Categorical Columns
le = LabelEncoder()
for col in cat_cols:
    amazon_df[col] = le.fit_transform(amazon_df[col])
print("- Label Encoding applied to categorical columns")

# Step 8: Scale Numeric Columns
scaler = StandardScaler()
scale_cols = ['stars', 'reviews', 'price', 'Price_per_Review', 'Log_Price']
amazon_df[scale_cols] = scaler.fit_transform(amazon_df[scale_cols])
print("- Standard Scaling applied to selected numeric columns")

# Step 9: Exploratory Data Analysis
print("\n📊 Exploratory Data Analysis:")

plt.figure(figsize=(12, 8))
sns.heatmap(amazon_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (Amazon Data)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(amazon_df['Log_Price'], bins=30, kde=True)
plt.title("Distribution of Log-Transformed Price")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=amazon_df, x='isBestSeller', y='price')
plt.title("Price Distribution by Bestseller Status")
plt.tight_layout()
plt.show()

# Step 10: Save the Cleaned Dataset
output_path = "cleaned_amazon_15000.csv"
amazon_df.to_csv(output_path, index=False)
print(f"\n💾 Cleaned Amazon dataset saved to: {output_path}")
