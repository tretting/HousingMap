import geopandas as gpd
import pandas as pd
import numpy as np
import json
import pickle
import plotly.graph_objects as go
import statsmodels.api as sm
from shapely.geometry import Point

# -------------------------------
# Load GeoJSON with Computed Fields and Predictions
# -------------------------------
gdf = gpd.read_file("data/Parcels/preds.geojson")

# -------------------------------
# Compute Centroids If Not Present
# -------------------------------
if not {"centroid_x", "centroid_y"}.issubset(gdf.columns):
    gdf["centroid"] = gdf.geometry.centroid
    gdf["centroid_x"] = gdf["centroid"].apply(lambda p: p.x)
    gdf["centroid_y"] = gdf["centroid"].apply(lambda p: p.y)

# -------------------------------
# Enforcement Criteria (Optional)
# -------------------------------
enforce_zoning_areas = True
if enforce_zoning_areas:
    allowed_zones = ["R-1", "R1", "R1-6", "R1-9", "R1E", "R-E", "SFD", "PDZ", "LDR", "GR"]
    def zoning_allowed(z):
        if pd.isnull(z):
            return False
        z_up = z.upper()
        for zone in allowed_zones:
            if zone.upper() in z_up:
                return True
        return False
    # For parcels not meeting the zoning criteria, set predicted_prob to NA.
    gdf.loc[~gdf['ZONING'].apply(zoning_allowed), 'predicted_prob_kf'] = np.nan

enforce_COS = False
if enforce_COS:
    allowed_zipcodes = ["80901", "80902", "80903", "80904", "80905", "80906", "80907", "80908", "80909", "80910",
                        "80911", "80912", "80913", "80914", "80915", "80916", "80917", "80918", "80919", "80920",
                        "80921", "80922", "80923", "80924", "80925", "80926", "80927", "80928", "80929", "80930",
                        "80931", "80932", "80933", "80934", "80935", "80936", "80937", "80938", "80939", "80941",
                        "80942", "80943", "80944", "80945", "80946", "80947", "80949", "80950", "80951", "80960",
                        "80962", "80970", "80977", "80995", "80997"]
    gdf = gdf[gdf['ZIPCODE'].isin(allowed_zipcodes)]

# -------------------------------
# Sorting: Draw Lower Scores first, so highest predicted_prob are on top
# -------------------------------
# gdf = gdf.sort_values(by="predicted_prob", ascending=True)

# -------------------------------
# Set Marker Size and Hover Text Based on Predicted Probability
# -------------------------------
# Use a fixed marker size if predicted_prob exists, otherwise 0.
gdf['marker_size'] = gdf['predicted_prob_kf'].apply(lambda x: 5 if pd.notnull(x) else 0)
# Create hover text with four-digit rounding or leave blank if no data.
gdf['hover_text_kf'] = gdf['predicted_prob_kf'].apply(
    lambda x: f"Infill Score (Predicted): {x:.4f}" if pd.notnull(x) else ""
)

# -------------------------------
# Create a Separate DataFrame for Permitted ADU Parcels
# -------------------------------
# Assumes that PERMIT_SUBMITTED is 1 for permitted parcels.
permitted_gdf = gdf[gdf["PERMIT_SUBMITTED"] == 1]

# -------------------------------
# Load ZIP-code Boundaries for Context
# -------------------------------
zip_gdf = gpd.read_file("data/processed/COS_shape_data.geojson")
zip_geojson = json.loads(zip_gdf.to_json())

# Lock the predicted probability color scale (expected range is [0,1])
score_min = 0
score_max = np.max(gdf["predicted_prob_kf"])
print(f"Locked Predicted Probability range: {score_min} to {score_max}")

# -------------------------------
# Create Interactive Plotly Map
# -------------------------------
fig = go.Figure()

# Base layer: All parcels colored by predicted_prob.
fig.add_trace(go.Scattermapbox(
    lat = gdf["centroid_y"],
    lon = gdf["centroid_x"],
    mode = "markers",
    marker = dict(
        size = gdf["marker_size"],
        color = gdf["predicted_prob_kf"],
        colorscale = "Viridis",
        cmin = score_min,
        cmax = score_max,
        colorbar = dict(title = "Infill Score\n (k-fold)")
    ),
    text = gdf["hover_text_kf"],
    hoverinfo = "text"
))

fig.update_layout(
    mapbox = dict(
        accesstoken = "YOUR_MAPBOX_ACCESS_TOKEN",  # Replace with your actual Mapbox token
        style = "carto-positron",
        center = dict(
            lat = gdf["centroid_y"].mean(),
            lon = gdf["centroid_x"].mean()
        ),
        zoom = 10,
        layers = [
            dict(
                sourcetype = "geojson",
                source = zip_geojson,
                type = "line",
                color = "gray",
                line = dict(width = 1)
            )
        ]
    ),
    margin = dict(l = 0, r = 0, t = 0, b = 0),
    title = "Infill Potential Map (Predicted ADU Probability)"
)

# 5. Build a minimal GeoDataFrame of POINTs at the centroids
minimal = gpd.GeoDataFrame({
    "ZONING" : gdf["ZONING"],
    "ZIPCODE" : gdf["ZIPCODE"],
    "PERMIT_SUBMITTED" : gdf["PERMIT_SUBMITTED"],
    "predicted_prob_lg": gdf["predicted_prob_lg"],
    "predicted_prob_kf": gdf["predicted_prob_kf"],
    "marker_size":       gdf["marker_size"],
    "hover_text_lg":        gdf["hover_text_lg"],
    "hover_text_kf":        gdf["hover_text_kf"],
}, geometry=[Point(xy) for xy in zip(gdf.centroid_x, gdf.centroid_y)],
   crs=gdf.crs)

# 6. Write it out
minimal.to_file("data/Parcels/preds_lg_kf.geojson", driver="GeoJSON")

fig.show()
