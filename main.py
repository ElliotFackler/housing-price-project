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
median_lot_frontage = features['Lot Frontage'].median()
features['Lot Frontage'].fillna(median_lot_frontage, inplace=True)

median_garage_yr_blt = features['Garage Yr Blt'].median()
features['Garage Yr Blt'].fillna(median_garage_yr_blt, inplace=True)

mode_garage_finish = features['Garage Finish'].mode()[0]
features['Garage Finish'].fillna(mode_garage_finish, inplace=True)

mode_garage_cond = features['Garage Cond'].mode()[0]
features['Garage Cond'].fillna(mode_garage_cond, inplace=True)

mode_garage_qual = features['Garage Qual'].mode()[0]
features['Garage Qual'].fillna(mode_garage_qual, inplace=True)

mode_garage_type = features['Garage Type'].mode()[0]
features['Garage Type'].fillna(mode_garage_type, inplace=True)


print(f"\nShape of X (features): {features.shape}")
print(f"Shape of y (target): {target.shape}")

# Split the training data to validate the model
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Check for missing values
missing_values = features.isnull().sum().sort_values(ascending=False)
print("\nMissing values per feature:\n", missing_values[missing_values > 0])