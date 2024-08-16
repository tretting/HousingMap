from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd
import geopandas as gpd
import json

app = Flask(__name__)

# Load Colorado Springs shape data
COS_shape_data = gpd.read_file("data/processed/COS_shape_data.geojson")

# Load processed statewide data
annual_data = pd.read_csv("data/processed/annual_data.csv")
monthly_data = pd.read_csv("data/processed/monthly_data.csv")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    # Convert the GeoDataFrame to GeoJSON and return it as JSON
    data_geojson = COS_shape_data.to_json()
    return jsonify(json.loads(data_geojson))

@app.route('/metrics')
def metrics():
    annual_metrics = annual_data.columns[2:].tolist()  # Exclude ZIP Code and Year columns
    monthly_metrics = monthly_data.columns[2:].tolist()  # Exclude ZIP Code and Date columns
    return jsonify({'annual': annual_metrics, 'monthly': monthly_metrics})

@app.route('/update_data', methods=['POST'])
def update_data():
    request_data = request.json
    metric = request_data['metric']
    data_type = request_data['data_type']
    date_type = request_data['date_type']
    start_date = request_data['start_date']
    end_date = request_data.get('end_date')
    change_type = request_data.get('change_type', 'total')  # default to 'total' if not provided

    print(f"Metric: {metric}")
    print(f"Data Type: {data_type}")
    print(f"Date Type: {date_type}")
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Change Type: {change_type}")

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

            # Print the head of the DataFrame to the terminal
            print("Combined Data Head:\n", combined_data.head())

        else:
            response_data = start_data[['RegionName', metric]].dropna().to_dict(orient='records')

            # Print the head of the start_data DataFrame to the terminal
            print("Start Data Head:\n", start_data.head())

    else:
        start_data = monthly_data[monthly_data['datesubstr'] == start_date]
        print(f"Start Data for {start_date}: {start_data}")
        if end_date:
            end_data = monthly_data[monthly_data['datesubstr'] == end_date]
            print(f"End Data for {end_date}: {end_data}")
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

            # Print the head of the combined DataFrame to the terminal
            print("Combined Data Head:\n", combined_data.head())

        else:
            response_data = start_data[['RegionName', metric]].dropna().to_dict(orient='records')

            # Print the head of the start_data DataFrame to the terminal
            print("Start Data Head:\n", start_data.head())

    print(f"Response Data: {response_data}")
    return jsonify(response_data)


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
