
# Step 1: Import Required Libraries

import pandas as pd  # for working with data tables
import numpy as np  # for numerical operations
import matplotlib.pyplot as plt  # for creating graphs
import seaborn as sns  # for clear statistical visualizations
from sklearn.preprocessing import LabelEncoder, StandardScaler  # for transforming data


# Step 2: Load the dataset

try:
    # Load the customer segmentation dataset from your computer
    customer_df = pd.read_csv("/Users/ifeomaigbokwe/Downloads/customer segmentation dataset/Test.csv")
except Exception as e:
    # If the file is not found or can't be read, show an error and stop the script
    print("❌ Error loading file:", e)
    exit()


# Step 3: Preview the dataset

print("✅ Dataset Loaded Successfully")
print("📄 Column Names:", customer_df.columns.tolist())  # Show all column names
print("\n🔍 First Five Rows:")  # Show a quick preview of data
print(customer_df.head())


# Step 4: Check for missing values

print("\n🧹 Missing Values in Each Column:")
print(customer_df.isnull().sum())  # Count missing values for each column

# Step 5: Handle Missing Data

# Select numeric columns (e.g., Age, Family_Size)
num_cols = customer_df.select_dtypes(include=[np.number]).columns.tolist()

# Fill missing numeric values with the median of the column (less affected by outliers)
customer_df[num_cols] = customer_df[num_cols].fillna(customer_df[num_cols].median())

# Select text-based (categorical) columns (e.g., Gender, Profession)
cat_cols = customer_df.select_dtypes(include=['object']).columns.tolist()

# Fill missing categorical values with the most common value (mode)
for col in cat_cols:
    customer_df[col] = customer_df[col].fillna(customer_df[col].mode()[0])

# Step 6: Feature Engineering (Creating new columns)

print("\n🛠️ Feature Engineering:")

# Group customers into age groups
bins = [0, 18, 35, 55, 100]  # Age ranges
labels = ['Teen', 'Young Adult', 'Adult', 'Senior']  # Group names
customer_df['Age_Group'] = pd.cut(customer_df['Age'], bins=bins, labels=labels, right=False)
print("- Created 'Age_Group' from 'Age'")

# Assume Family_Size includes the individual; copy it to Total_Family
customer_df['Total_Family'] = customer_df['Family_Size']
print("- Created 'Total_Family' from 'Family_Size'")

# Create an interaction feature between Work Experience and Age (scaled)
# First scale the age so it fits into a smaller numerical range
scaler_age = StandardScaler()
customer_df['Age_Scaled'] = scaler_age.fit_transform(customer_df[['Age']])

# Multiply scaled age with work experience to create a new feature
customer_df['Work_Exp_Age_Interaction'] = customer_df['Work_Experience'] * customer_df['Age_Scaled']
print("- Created 'Work_Exp_Age_Interaction' from 'Work_Experience' and 'Age' (scaled)")

# Step 7: Encode Categorical Columns

# Convert all text-based categories (e.g., 'Male', 'Engineer') into numbers
# This also includes the new column 'Age_Group'
le = LabelEncoder()
for col in cat_cols + ['Age_Group']:  # Include Age_Group for encoding
    customer_df[col] = le.fit_transform(customer_df[col])
print("- Label Encoding applied to categorical columns (including 'Age_Group')")

# Step 8: Scale Numeric Columns

# Scale all numeric columns (except ID)
scaler = StandardScaler()

# Avoid scaling ID, but include engineered features
numerical_cols_to_scale = [col for col in num_cols if col not in ['ID']]

# Scale the selected numeric features + new engineered features
customer_df[numerical_cols_to_scale + ['Age_Scaled', 'Work_Exp_Age_Interaction', 'Total_Family']] = scaler.fit_transform(
    customer_df[numerical_cols_to_scale + ['Age_Scaled', 'Work_Exp_Age_Interaction', 'Total_Family']]
)
print("- Standard Scaling applied to numerical columns")

# Step 9: Exploratory Data Analysis (EDA)

print("\n📊 Exploratory Data Analysis (including new features):")

# Show relationships between numerical columns using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(customer_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (including engineered features)")
plt.tight_layout()
plt.show()

# Count how many people are in each Age Group
plt.figure(figsize=(8, 5))
sns.countplot(data=customer_df, x='Age_Group')
plt.title("Distribution of Customer Age Groups")
plt.xlabel("Age Group")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# Boxplot to see spending patterns across age groups
plt.figure(figsize=(8, 6))
sns.boxplot(data=customer_df, x='Age_Group', y='Spending_Score')
plt.title("Spending Score Distribution by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Spending Score (Scaled)")
plt.tight_layout()
plt.show()

# If segmentation labels exist, plot how many customers are in each group
if 'Segmentation' in customer_df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=customer_df, x='Segmentation')
    plt.title("Distribution of Customer Segments")
    plt.xlabel("Segment")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.show()

# Step 10: Save the Final Cleaned Dataset

# Save the cleaned and transformed dataset to a CSV file for future use
output_path = "cleaned_customer_segmentation_engineered.csv"
customer_df.to_csv(output_path, index=False)
print(f"\n💾 Cleaned dataset with engineered features saved to: {output_path}")
