import geopandas as gpd
import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
import json
import plotly.graph_objects as go
import pickle
import os

# Ensure the output directory exists
output_dir = "data/trained"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Data Loading and Cleaning
# -------------------------------
# Load parcel data GeoJSON
parcel_gdf = gpd.read_file("data/Parcels/Updated_parcels_permits.geojson")

# We'll use the three normalized features:
features = parcel_gdf[['norm_equity', 'inv_lotcov', 'norm_land_value']].copy()
target = parcel_gdf['PERMIT_SUBMITTED']

# Drop rows where any of these features are missing.
data = pd.concat([features, target], axis=1).dropna()
features_clean = data[['norm_equity', 'inv_lotcov', 'norm_land_value']]
target_clean = data['PERMIT_SUBMITTED']

# Add a constant term for the intercept.
features_clean = sm.add_constant(features_clean)

# Fit a logistic regression model.
logit_model = sm.Logit(target_clean, features_clean)
result = logit_model.fit()

# Print the summary of the logistic regression model.
print(result.summary())

# Optionally, calculate the predicted probability for each row in the training subset:
data['predicted_prob'] = result.predict(features_clean)

# You can now inspect the coefficients:
coefficients = result.params
print("\nLearned Coefficients:")
print(coefficients)

# -------------------------------
# Save the Fitted Logistic Model Using Pickle
# -------------------------------
model_filepath = os.path.join(output_dir, "logistic_model.pkl")
with open(model_filepath, "wb") as f:
    pickle.dump(result, f)
print(f"Logistic model saved to '{model_filepath}'.")

# -------------------------------
# Compute Predicted Probabilities for All Parcels
# -------------------------------
# Prepare the feature matrix for every parcel.
X_full = parcel_gdf[['norm_equity', 'inv_lotcov', 'norm_land_value']]
X_full_const = sm.add_constant(X_full)
parcel_gdf["predicted_prob_lg"] = result.predict(X_full_const)

# -------------------------------
# Save the Updated GeoJSON File (with Predictions)
# -------------------------------
output_geojson = "data/Parcels/Updated_parcels_permits_with_preds.geojson"
parcel_gdf.to_file(output_geojson, driver="GeoJSON")
print(f"Updated GeoJSON with predictions saved to '{output_geojson}'.")

# Print summary statistics of the non-NA predicted infill scores
summary_stats = parcel_gdf["predicted_prob_lg"].describe()
print("Summary statistics for predicted infill scores (non-NA values):")
print(summary_stats)

