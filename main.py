import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv('AmesHousing.csv')

num_rows = data.shape[0]
print(f"\nNumber of rows in the CSV: {num_rows}")

# Create a DataFrame for the features and the target and drop unnecessary columns.
target = data['SalePrice']
features = data.drop(columns=['SalePrice', 'Id', 'Misc Feature', 'Pool QC', 'Alley', 'Fence', 'Mas Vnr Type', 'Fireplace Qu'], errors='ignore')

# Fill missing values
for col in features.columns:
    if features[col].isnull().sum() > 0:
        if features[col].dtype in [np.float64, np.int64]:
            median = features[col].median()
            features[col].fillna(median, inplace=True)
            print(f"Filled missing values in '{col}' with median: {median}")
        else:
            mode = features[col].mode()[0]
            features[col].fillna(mode, inplace=True)
            print(f"Filled missing values in '{col}' with mode: {mode}")


print(f"\nShape of X (features): {features.shape}")
print(f"Shape of y (target): {target.shape}")

# Split the training data to validate the model
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Check for missing values
missing_values = features.isnull().sum().sort_values(ascending=False)
print("\nMissing values per feature:\n", missing_values[missing_values > 0])