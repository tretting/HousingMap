from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
import geopandas as gpd
import json

app = Flask(__name__)


# Load shape and regional data - these are the shapefile data of zipcodes
# from the TIGER database that roughly correspond to most recent zipcode area
# coverage, along with the data compiled through my other research. This data is
# zipcode-level, monthly home value data and annual socioeconomic data from the census
COS_shape_data = gpd.read_file("data/processed/COS_shape_data.geojson")
annual_data = pd.read_csv("data/processed/annual_data.csv")
monthly_data = pd.read_csv("data/processed/monthly_data.csv")

# Load forecast datasets - these are the SARIMA and LSTM models forecasted
# from monthly ZHVI data Jan 2011 through Nov 2024
sarima_data = pd.read_csv("data/forecast_data/SARIMA_ZHVI_forecast_Colorado_Springs.csv")
lstm_data = pd.read_csv("data/forecast_data/LSTM_ZHVI_forecast_Colorado_Springs.csv")

# Rename forecast column for clarity if needed
sarima_data.rename(columns={"Forecast_ZHVI": "SARIMA_forecast"}, inplace=True)
lstm_data.rename(columns={"Forecast_ZHVI": "LSTM_forecast"}, inplace=True)

# Ensure date columns are consistent (we will use YYYY-MM-DD as string format)
# Optionally, you can convert date strings to datetime objects if needed.

# Load additional data - this is data that aggregates county treasurer data of
# individual plot coverage to the zipcode level
plat_coverage = pd.read_csv('data/processed/plat_coverage.csv')
zipcode_coverage = pd.read_csv('data/processed/zipcode_coverage.csv')

# List of ZIP codes for Colorado Springs
COS_zips = ["80901", "80902", "80903", "80904", "80905", "80906", "80907", "80908", "80909", "80910",
            "80911", "80912", "80913", "80914", "80915", "80916", "80917", "80918", "80919", "80920",
            "80921", "80922", "80923", "80924", "80925", "80926", "80927", "80928", "80929", "80930",
            "80931", "80932", "80933", "80934", "80935", "80936", "80937", "80938", "80939", "80941",
            "80942", "80943", "80944", "80945", "80946", "80947", "80949", "80950", "80951", "80960",
            "80962", "80970", "80977", "80995", "80997"]

# Ensure ZIPCODE is formatted correctly
zipcode_coverage = zipcode_coverage.dropna(subset=['ZIPCODE']).reset_index(drop=True)
zipcode_coverage['ZIPCODE'] = zipcode_coverage['ZIPCODE'].astype(int).astype(str).str.zfill(5)
zipcode_coverage = zipcode_coverage[zipcode_coverage['ZIPCODE'].isin(COS_zips)].reset_index(drop=True)

# ---------------------------
# Routes
# ---------------------------

# Landing page
@app.route('/')
def index():
    return render_template('index.html')

# Loading shape data for zipcode mapping
@app.route('/data')
def data():
    data_geojson = COS_shape_data.to_json()
    return jsonify(json.loads(data_geojson))

# Loading pull-down selectable data
@app.route('/metrics')
def metrics():
    annual_metrics = annual_data.columns[2:].tolist()
    monthly_metrics = monthly_data.columns[2:].tolist()
    custom_metrics = ['LotCoverage']
    # Hard-code forecast metric names (they should match those used in update_data)
    forecast_metrics = ['SARIMA_forecast', 'LSTM_forecast']
    return jsonify({
        'annual': annual_metrics,
        'monthly': monthly_metrics,
        'custom': custom_metrics,
        'forecast': forecast_metrics
    })

@app.route('/lot_coverage_zipcode')
def lot_coverage_zipcode():
    return jsonify(zipcode_coverage.to_dict(orient='records'))

# Loading and formatting forecasting data
@app.route('/forecast_dates')
def forecast_dates():
    # Combine unique forecast dates from both datasets
    sarima_dates = sarima_data['Date'].unique().tolist()
    lstm_dates = lstm_data['Date'].unique().tolist()
    all_dates = sorted(set(sarima_dates + lstm_dates))
    return jsonify(all_dates)

# Rading page changes and updating calls accordingly
@app.route('/update_data', methods=['POST'])
def update_data():
    request_data = request.json
    metric = request_data.get('metric')
    data_type = request_data.get('data_type')
    date_type = request_data.get('date_type')
    start_date = request_data.get('start_date')
    end_date = request_data.get('end_date')
    change_type = request_data.get('change_type', 'total')

    # Handle forecast data separately
    if date_type == 'forecast':
        # Select appropriate DataFrame based on metric
        if metric == 'SARIMA_forecast':
            df = sarima_data
            value_col = 'SARIMA_forecast'
        elif metric == 'LSTM_forecast':
            df = lstm_data
            value_col = 'LSTM_forecast'
        else:
            return jsonify([])

        # Filter forecast data for the selected date
        filtered = df[df['Date'] == start_date]
        response_data = []
        for _, row in filtered.iterrows():
            response_data.append({
                'RegionName': str(row['ZIP']).zfill(5),  # Ensure ZIP code format matches
                metric: row[value_col]
            })
        return jsonify(response_data)

    # Logic for annual, monthly, and custom metrics in the pulldown
    if date_type == 'annual':
        start_data = annual_data[annual_data['year'] == int(start_date)]
        if end_date:
            end_data = annual_data[annual_data['year'] == int(end_date)]
            start_data = start_data.set_index('RegionName')
            end_data = end_data.set_index('RegionName')
            start_data = start_data.add_suffix('_start')
            end_data = end_data.add_suffix('_end')
            combined_data = start_data.join(end_data)
            if change_type == 'percent':
                combined_data[metric] = ((combined_data[f"{metric}_end"] - combined_data[f"{metric}_start"]) / combined_data[f"{metric}_start"]) * 100
                combined_data[metric] = combined_data[metric].round(1)
            else:
                combined_data[metric] = combined_data[f"{metric}_end"] - combined_data[f"{metric}_start"]
                combined_data[metric] = combined_data[metric].round(0)
            combined_data = combined_data.dropna(subset=[metric])
            response_data = combined_data[[metric]].reset_index().to_dict(orient='records')
        else:
            response_data = start_data[['RegionName', metric]].dropna().to_dict(orient='records')
        return jsonify(response_data)

    else:
        # Otherwise we have monthly and custom data
        start_data = monthly_data[monthly_data['datesubstr'] == start_date]
        if end_date:
            end_data = monthly_data[monthly_data['datesubstr'] == end_date]
            start_data = start_data.set_index('RegionName')
            end_data = end_data.set_index('RegionName')
            start_data = start_data.add_suffix('_start')
            end_data = end_data.add_suffix('_end')
            combined_data = start_data.join(end_data)
            if change_type == 'percent':
                combined_data[metric] = ((combined_data[f"{metric}_end"] - combined_data[f"{metric}_start"]) / combined_data[f"{metric}_start"]) * 100
                combined_data[metric] = combined_data[metric].round(1)
            else:
                combined_data[metric] = combined_data[f"{metric}_end"] - combined_data[f"{metric}_start"]
                combined_data[metric] = combined_data[metric].round(0)
            combined_data = combined_data.dropna(subset=[metric])
            response_data = combined_data[[metric]].reset_index().to_dict(orient='records')
        else:
            response_data = start_data[['RegionName', metric]].dropna().to_dict(orient='records')
        return jsonify(response_data)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
