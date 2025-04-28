import geopandas as gpd
import pandas as pd
import numpy as np
import re
import statsmodels.api as sm   # (Note: Not used in the scikit-learn model)
import json
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Ensure the output directory exists
output_dir = "data/trained"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Data Loading and Cleaning
# -------------------------------
# Load parcel data GeoJSON
parcel_gdf = gpd.read_file("data/Parcels/Updated_parcels_permits_with_preds.geojson")

# We assume the following three normalized features exist in the file:
required_features = ['norm_equity', 'inv_lotcov', 'norm_land_value']
for col in required_features:
    if col not in parcel_gdf.columns:
        raise ValueError(f"Required column '{col}' not found in the GeoJSON file.")

# The target is assumed to be in 'PERMIT_SUBMITTED'
if 'PERMIT_SUBMITTED' not in parcel_gdf.columns:
    raise ValueError("Target column 'PERMIT_SUBMITTED' not found in the GeoJSON file.")

# Extract features and target
features = parcel_gdf[required_features].copy()
target = parcel_gdf['PERMIT_SUBMITTED']

# Create a combined DataFrame and drop rows with missing features
data = pd.concat([features, target], axis=1).dropna()
features_clean = data[required_features]
target_clean = data['PERMIT_SUBMITTED']

# Convert features and target to numpy arrays for cross-validation
X = features_clean.values
y = target_clean.values

# -------------------------------
# K-fold Cross Validation
# -------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(solver='liblinear')  # You can adjust regularization if needed.
scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
print("Mean AUC:", np.mean(scores))

# -------------------------------
# Train Final Model on Full Dataset
# -------------------------------
model.fit(X, y)

# -------------------------------
# Print Model Coefficients
# -------------------------------
print("\nLearned Coefficients:")
print("Intercept:", model.intercept_[0])
coef_dict = {feature: coef for feature, coef in zip(features_clean.columns, model.coef_[0])}
print(coef_dict)

# -------------------------------
# Save the Fitted Logistic Model Using Pickle
# -------------------------------
model_filepath = os.path.join(output_dir, "kfoldcross_model1.pkl")
with open(model_filepath, "wb") as f:
    pickle.dump(model, f)
print(f"K-fold model saved to '{model_filepath}'.")

# -------------------------------
# Compute Predicted Probabilities for All Parcels
# -------------------------------
# We want predictions only for parcels with non-missing required features.
# Create a copy of the full GeoDataFrame.
gdf_full = parcel_gdf.copy()

# Identify indices where all required features are non-null.
valid_idx = gdf_full[required_features].dropna().index
# For these rows, extract the features as a numpy array.
X_full = gdf_full.loc[valid_idx, required_features].values

# Compute predicted probabilities using the fitted model.
gdf_full.loc[valid_idx, "predicted_prob_kf"] = model.predict_proba(X_full)[:, 1]
# For rows missing any of the required features, assign NA.
gdf_full["predicted_prob_kf"] = gdf_full["predicted_prob_kf"].astype(float)

# -------------------------------
# Save the Updated GeoJSON File (with Predictions)
# -------------------------------
output_geojson = "data/Parcels/Updated_parcels_permits_with_preds_kfold1.geojson"
gdf_full.to_file(output_geojson, driver="GeoJSON")
print(f"Updated GeoJSON with k-fold predictions saved to '{output_geojson}'.")

# -------------------------------
# Print Summary Statistics for Predicted Infill Scores (non-NA values)
# -------------------------------
summary_stats = gdf_full["predicted_prob_kf"].dropna().describe()
print("Summary statistics for predicted infill scores (non-NA values):")
print(summary_stats)
