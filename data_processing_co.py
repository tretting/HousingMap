import pandas as pd
import geopandas as gpd
import pyreadr
import json

def read_rds_file(file_path):
    result = pyreadr.read_r(file_path)
    df = next(iter(result.values()))
    return df

def sanitize_column_name(col_name):
    return col_name.replace(' ', '_').replace('(', '').replace(')', '').replace('>', 'gt')

def process_data():
    # Path to shapefile
    dir_tiger = "data/tiger/tl_2023_us_zcta520"
    tiger_filename = "tl_2023_us_zcta520.shp"
    dir_RDS = "data/demo_and_housing/"
    RDS_filename = "zhvi_demographic_043024.RDS"
    shapedata = gpd.read_file(f"{dir_tiger}/{tiger_filename}")

    # Filter the ZCTA data for Colorado ZIP codes (Assuming Colorado's state FIPS code is 08)
    shapedata['state'] = shapedata['GEOID20'].str[:2]
    CO_shape_data = shapedata[shapedata['state'] == "08"].copy()
    CO_shape_data['GEOID20'] = CO_shape_data['GEOID20'].astype(str)

    # Save the shape data to a separate GeoJSON file
    CO_shape_data.to_file("data/processed/CO_shape_data.geojson", driver="GeoJSON")

    # Read your RDS data
    full_data = read_rds_file(f"{dir_RDS}/{RDS_filename}")

    # Filter the data for Colorado ZIP codes
    CO_zips = CO_shape_data['GEOID20'].unique().tolist()
    filtered_data = full_data[full_data['RegionName'].isin(CO_zips)].copy()
    filtered_data['RegionName'] = filtered_data['RegionName'].astype(str)

    # Define the columns that are annual and monthly
    annual_columns = ['RegionName', 'year', 'Total_Households', 'Pct_200k', 'Re_med_inc', 'Re_mean_inc', 
                      'Tot_Pop_16', 'LFP_Pop_16', 'EPR_Pop_16', 'UER_Pop_16', 'avg_comm']
    
    monthly_columns = ['RegionName', 'datesubstr', 'ZHVI_sfr', 'ZHVI_all', 'ZHVI_1br', 'ZHVI_2br', 
                       'ZHVI_3br', 'ZHVI_4br', 'ZHVI_5br', 'ZHVI_con', 'ZHVI_sfr_CL', 'ZHVI_all_CL', 
                       'ZHVI_1br_CL', 'ZHVI_2br_CL', 'ZHVI_3br_CL', 'ZHVI_4br_CL', 'ZHVI_5br_CL', 'ZHVI_con_CL']

    # Split the data into annual and monthly datasets
    annual_data = filtered_data[annual_columns]
    monthly_data = filtered_data[monthly_columns]

    # Create sanitized column names
    sanitized_annual_columns = [sanitize_column_name(col) for col in annual_columns]
    sanitized_monthly_columns = [sanitize_column_name(col) for col in monthly_columns]

    # Rename columns for better user understanding
    annual_data.columns = sanitized_annual_columns
    monthly_data.columns = sanitized_monthly_columns
    
    # Save the processed data to CSV files
    annual_data.to_csv('data/processed/annual_data_CO.csv', index=False)
    monthly_data.to_csv('data/processed/monthly_data_CO.csv', index=False)

if __name__ == '__main__':
    process_data()
