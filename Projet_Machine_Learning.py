# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 09:34:43 2025

@author: Jmaro
"""
# Cell 1
# List of essential libraries (Retrieve data from CSV files + plot curves)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Plot the map (Visualization)
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point

# To digitize data
from sklearn.preprocessing import LabelEncoder

# For the prediction model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# To calculate errors
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# To test other models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor


# Cell 2

# The essentials
!pip install pandas
!pip install seaborn
!pip install matplotlib
!pip install numpy

# For the map
!pip install geopandas
!pip install contextily
!pip install shapely

# For the machin learning
!pip install scikit-learn
!pip install xgboost
!pip install lightgbm

# Cell 3
df = pd.read_csv('AB_NYC_2019.csv')
df.head()

# Cell 4
print(df)

# Cell 5
df.columns

# Cell 6
df.info()

# Cell 7
df.describe()

# Cell 8
df.shape

# Cell 9
# Clean the data (keep only valid points)
df_map = df.dropna(subset=['latitude', 'longitude', 'price']).copy()

# Create geometric objects (points)
geometry = [Point(xy) for xy in zip(df_map["longitude"], df_map["latitude"])]
gdf = gpd.GeoDataFrame(df_map, geometry=geometry, crs="EPSG:4326")  # GPS coordinates

# Convert to web projection (Web Mercator)
gdf = gdf.to_crs(epsg=3857)

# Create the figure
fig, ax = plt.subplots(figsize=(12, 10))

# Scatter plot of points based on their log_price
gdf.plot(
    ax=ax,
    column="price",
    cmap="YlOrRd",
    markersize=5,
    legend=True,
    alpha=0.7,
    legend_kwds={
        'shrink': 0.23,         # Reduce the colorbar height
        'label': "price",   # Colorbar title
        'orientation': "vertical",
        'pad': 0.01,            # Space between the map and the colorbar
    }
)

# Add basemap (OpenStreetMap)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Clean up the rendering
ax.set_axis_off()
plt.title("Geographical distribution of prices (price)", fontsize=15)
plt.tight_layout()
plt.show()


# Cell 10
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df, 
    x='neighbourhood_group', 
    y='price', 
    palette='Set2',
    showfliers=False  # remove extreme values for see better the box
)

plt.title("Price by type of reservation", fontsize=16)
plt.xlabel("Type of reservation", fontsize=12)
plt.ylabel("Price in dollars", fontsize=12)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Cell 11
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df, 
    x='room_type', 
    y='price', 
    palette='Set2',
    showfliers=False  # retire les valeurs extrêmes pour mieux voir la boîte
)

plt.title("Price by type of reservation", fontsize=16)
plt.xlabel("type of reservation", fontsize=12)
plt.ylabel("Price en dollars", fontsize=12)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# Cell 12
sns.histplot(df['price'], kde=True, bins=30)
plt.title("Distribution of the price")
plt.xlabel("Price in dollars")
plt.ylabel("Frequence")
plt.show()

# Cell 13
plt.figure(figsize=(10,5))
plt.scatter(df.index, df['price'], alpha=0.5)
plt.title("Scatter plot of the price")
plt.xlabel("Index")
plt.ylabel("Price")
plt.show()

# Cell 14
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna('Missing')
    freq = df[col].value_counts(normalize=True)
    df[col] = df[col].map(freq)

# Cell 15
df.head()

# Cell 16
df = df[df['price'] > 0]

# Cell 17
df.info()

# Cell 18
df.shape

# Cell 19
df = df.dropna()


# Cell 20
df.shape

# Cell 21
df['price_log'] = np.log1p(df['price'])
df = df.drop(columns=['price'])


# Cell 22
# Clean the data (keep only valid points)
df_map = df.dropna(subset=['latitude', 'longitude', 'price_log']).copy()

# Create geometric objects (points)
geometry = [Point(xy) for xy in zip(df_map["longitude"], df_map["latitude"])]
gdf = gpd.GeoDataFrame(df_map, geometry=geometry, crs="EPSG:4326")  # GPS coordinates

# Convert to web projection (Web Mercator)
gdf = gdf.to_crs(epsg=3857)

# Create the figure
fig, ax = plt.subplots(figsize=(12, 10))

# Scatter plot of points based on their log_price
gdf.plot(
    ax=ax,
    column="price_log",
    cmap="YlOrRd",
    markersize=5,
    legend=True,
    alpha=0.7,
    legend_kwds={
        'shrink': 0.23,         # Reduce the colorbar height
        'label': "price_log",   # Colorbar title
        'orientation': "vertical",
        'pad': 0.01,            # Space between the map and the colorbar
    }
)

# Add basemap (OpenStreetMap)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Clean up the rendering
ax.set_axis_off()
plt.title("Geographical distribution of prices (price_log)", fontsize=15)
plt.tight_layout()
plt.show()


# Cell 23
sns.histplot(df['price_log'], kde=True, bins=30)
plt.title("Distribution of log price")
plt.xlabel("log_price")
plt.ylabel("Frequence")
plt.show()

# Cell 24
# List of columns to remove
columns_to_drop = ['latitude', 'longitude', 'host_id']

# Drop them in place
df.drop(columns=columns_to_drop, inplace=True)
print(df)



# Cell 25
# Compute the correlation matrix (numeric columns only)
correlation_matrix = df.corr(numeric_only=True)

# Extract the correlation with log_price, removing log_price itself
log_price_corr = correlation_matrix['price_log'].drop('price_log')

# Sort the values in descending order
log_price_corr_sorted = log_price_corr.sort_values(ascending=False)

# Display a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=log_price_corr_sorted.values, y=log_price_corr_sorted.index, palette='coolwarm')
plt.title('Correlation of variables with price')
plt.xlabel('Correlation')
plt.ylabel('Variables')
plt.tight_layout()
plt.show()


# Cell 26
# List of columns to remove
columns_to_drop = ['neighbourhood', 'name']

# Drop them in place
df.drop(columns=columns_to_drop, inplace=True)
print(df)



# Cell 27
y = df['price_log']
X = df.drop(columns=['price_log'])  # Remove 'price_log' from the features

# Train/test split(80/20) for validation 
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and categorical columns 
numeric_features = [col for col in X_train.columns if pd.api.types.is_numeric_dtype(X_train[col])]
categorical_features = [col for col in X_train.columns if pd.api.types.is_string_dtype(X_train[col])]

# Pipelines 
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline with XGBRegressor 
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, random_state=42))
])

# Training 
model.fit(X_train, y_train)

# Predictions on validation set 
y_valid_pred = model.predict(X_valid)

# Save predictions 
pd.DataFrame({'price_log': y_valid_pred}).to_csv('predictions.csv', index=False)
print("Predictions saved in predictions.csv")


# Cell 28
df_final = pd.read_csv("predictions.csv")
print(df_final)

# Cell 29
sns.histplot(df_final['price_log'], kde=True, bins=30)
plt.title("Distribution of price_log")
plt.xlabel("log_price")
plt.ylabel("Frequency")
plt.show()

# Cell 30
# Predictions on validation set
residuals = y_valid - y_valid_pred

sns.histplot(residuals, kde=True)
plt.title("Distribution of residuals (errors)")
plt.xlabel("log_price - prediction")
plt.show()

# Cell 31
rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
print(f"Root Mean Squared Error (RMSE) sur train : {rmse:.3f}")

# Cell 32
print(f"Validation score: {r2_score(y_true=y_valid, y_pred=y_valid_pred)}")

# Cell 33
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_valid, y=y_valid_pred, alpha=0.5)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], color='red', linestyle='--')
plt.xlabel("Real values (price_log)")
plt.ylabel("Predicted values")
plt.title("Real vs predicted")
plt.grid(True)
plt.show()

# Cell 34
# Target and features
y_train = df['price_log']
X_train = df.drop(columns=['price_log'])
X_test = df.copy()

# Split 80% for training and 20% for validation
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Column preparation
numeric_features = [col for col in X_train.columns if pd.api.types.is_numeric_dtype(X_train[col])]
categorical_features = [col for col in X_train.columns if pd.api.types.is_string_dtype(X_train[col])]

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# List of models to test
models = {
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Ridge': Ridge(alpha=1.0),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=100, random_state=42)
}

# Model evaluation
results = []

for name, regressor in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    pipeline.fit(X_train, y_train)
    y_valid_pred = pipeline.predict(X_train)
    
    r2 = r2_score(y_train, y_valid_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_valid_pred))
    
    results.append({
        'Model': name,
        'R² (train)': round(r2, 4),
        'RMSE (train)': round(rmse, 4)
    })

# Display the comparison table
results_df = pd.DataFrame(results).sort_values(by='R² (train)', ascending=False)
print(results_df)



# Cell 35
#  Detect numeric and categorical columns
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

#  Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

#  Combine everything in a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

#  Full pipeline with RandomForest
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

#  Train the model
pipeline.fit(X_train, y_train)

#  Predict on X_test
y_pred = pipeline.predict(X_test)

#  Create the output CSV file
df_final = pd.DataFrame({'id': id, 'price_log': y_pred})
df_final.to_csv('predictions_random_forest.csv', index=False)

print("Predictions saved in predictions_random_forest.csv")

# Cell 36
# Predictions on validation set
y_valid_pred = pipeline.predict(X_valid)
residuals = y_valid - y_valid_pred

# Residual distribution
sns.histplot(residuals, kde=True)
plt.title("Distribution of residuals (errors)")
plt.xlabel("price_log - prediction")
plt.show()

# Real vs Predicted scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_valid, y=y_valid_pred, alpha=0.5)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], color='red', linestyle='--')
plt.xlabel("Actual values (price_log)")
plt.ylabel("Predicted values")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()

# Cell 37
df_final_test = pd.DataFrame({'price_log': y_valid_pred})
print(df_final_test)


