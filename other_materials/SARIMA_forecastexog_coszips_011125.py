# =============================================================================
# This code performs SARIMA forecasting of Zillow Home Value Index (ZHVI) for
# single-family residential (sfr) data in Colorado Springs. It uses historical 
# data, covariate projections for future economic factors (like labor force 
# participation and median income), and outputs both forecasts and 
# performance metrics (MAE, MSE, RMSE).
#
# Comments have been added throughout to clarify the workflow, 
# methods, and logic for a more general audience. Please note that this code is
# included for completeness and verifying the forecasting methodology. The data
# is provided but you will need to ensure your directory is correctly
# structured, otherwise the code will fail to compile.
# =============================================================================

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress ConvergenceWarnings from statsmodels (they occur if optimization doesn't converge easily)
warnings.filterwarnings("ignore", category=UserWarning, message="Maximum Likelihood optimization failed to converge")

# =============================================================================
# 1) DATA LOADING AND FILTERING
# =============================================================================

# Load the CSV data file containing Zillow Home Value Index (ZHVI) and covariates
file_path = "zhvi_allbr_MLdata_010825.csv"
df = pd.read_csv(file_path)

# Convert the Date column to a valid datetime format for time-series operations
df['Date'] = pd.to_datetime(df['Date'])

# Filter the DataFrame to include only Colorado Springs data from 2011-01-01 onward
cs_data = df[(df['City'] == "Colorado Springs") & (df['Date'] >= '2011-01-01')].copy()

# Collect all the ZIP codes available for Colorado Springs
zip_codes = cs_data['RegionName'].unique()

# Define which covariates (exogenous variables) we want to use in the model
covariates = ['LFP_Pop_16', 'Nom_med_inc']

# Prepare an interactive Plotly figure to later display historical vs forecasted data
fig = go.Figure()

# Initialize a list to store forecast results for all ZIP codes
forecast_results = []

# =============================================================================
# 2) MAIN LOOP OVER EACH ZIP CODE
# =============================================================================

for zip_code in zip_codes:
    # Extract data for one particular ZIP code, sorted by Date
    data_zip = cs_data[cs_data['RegionName'] == zip_code].copy()
    data_zip = data_zip.sort_values(by='Date')
    
    # Skip if there's no data for this ZIP code
    if data_zip.empty:
        continue

    # Apply a log-transform to the ZHVI values. This often helps stabilize variance.
    data_zip['log_ZHVI_sfr'] = np.log(data_zip['ZHVI_sfr'])
    
    # Replace infinite values resulting from the log transform and remove rows with nulls
    data_zip.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_zip.dropna(subset=['log_ZHVI_sfr'], inplace=True)

    # Convert the ZHVI column into a monthly (ME) time series, dropping any missing values
    ts = data_zip.set_index('Date')['log_ZHVI_sfr'].asfreq('ME').dropna()

    # We need at least 36 data points (3 years) to perform backtesting reliably
    if len(ts) < 36:
        continue

    # Create a monthly version of the exogenous covariates. We use forward fill to handle missing data.
    data_cov = data_zip.set_index('Date')[covariates].resample('ME').ffill()
    
    # Also create a yearly resampled version of these covariates to build simple linear models for future extrapolation
    yearly = data_zip.set_index('Date')[covariates].resample('YE').mean()
    
    # Add a 'time' column to represent the sequential years, so we can fit a linear model
    yearly['time'] = np.arange(len(yearly))

    # =============================================================================
    # 2a) BUILD SIMPLE LINEAR MODELS TO PROJECT FUTURE COVARIATES (2024, 2025)
    # =============================================================================

    # Dictionary to store future predicted values of each covariate by year
    future_predictions = {cov: {} for cov in covariates}

    for cov in covariates:
        y = yearly[cov]               # The yearly values for the covariate
        X = sm.add_constant(yearly['time'])  # Simple linear regression with an intercept and a time trend
        valid = y.notna()             # Make sure we're only using valid data
        y = y[valid]
        X = X[valid]

        # If there's insufficient data or no variation, just use the most recent value
        if len(y) < 2 or y.nunique() == 1:
            last_value = yearly[cov].iloc[-1] if not yearly[cov].empty else np.nan
            future_predictions[cov]['2024'] = last_value
            future_predictions[cov]['2025'] = last_value
            continue

        # Fit a simple linear model to predict how the covariate might trend over time
        model_cov = sm.OLS(y, X).fit()

        # Predict for next two time points: e.g., len(yearly) (2024) and len(yearly)+1 (2025)
        next_times = [len(yearly), len(yearly) + 1]
        preds = model_cov.predict(sm.add_constant(next_times))
        
        future_predictions[cov]['2024'] = preds[0]
        future_predictions[cov]['2025'] = preds[1]

    # =============================================================================
    # 2b) SET UP FORECAST PARAMS: HOW FAR AHEAD WE WANT TO FORECAST
    # =============================================================================

    forecast_steps = 12  # We'll forecast 12 months into the future
    
    # The forecast starts the month after the last known data point
    forecast_start = ts.index[-1] + pd.offsets.MonthBegin(1)
    
    # Create a date range for the forecast period
    forecast_index = pd.date_range(start=forecast_start, periods=forecast_steps, freq='ME')

    # Prepare a DataFrame that will hold future exogenous values (for 12 months ahead)
    future_exog = pd.DataFrame(index=forecast_index, columns=covariates)

    # Populate this future exogenous DataFrame using our yearly predictions and/or the last known data
    for date in forecast_index:
        year = date.year
        for cov in covariates:
            if year == 2024:
                # If the forecasted year is 2024, use that predicted value
                future_exog.loc[date, cov] = future_predictions[cov].get('2024', np.nan)
            elif year == 2025:
                # If the forecasted year is 2025, use that predicted value
                future_exog.loc[date, cov] = future_predictions[cov].get('2025', np.nan)
            else:
                # Otherwise, just hold the covariate at its most recent known level
                future_exog.loc[date, cov] = data_cov[cov].iloc[-1] if not data_cov[cov].empty else np.nan

    # We're also inserting rows at the end of each future year (2024, 2025) into the extended DataFrame
    rows_to_add = []
    for cov in covariates:
        for yr in ['2024', '2025']:
            rows_to_add.append({'Date': pd.Timestamp(f'{yr}-12-31'), cov: future_predictions[cov].get(yr, np.nan)})
    new_rows_df = pd.DataFrame(rows_to_add)

    # Concatenate the new rows with our original ZIP-specific data
    data_extended = pd.concat([data_zip, new_rows_df], ignore_index=True)
    data_extended.sort_values(by='Date', inplace=True)
    
    # Group by Date (avoiding duplicates) and ensure it's monthly resampled with forward-fill for the covariates
    data_extended = data_extended.groupby('Date', as_index=False).first()
    data_cov_extended = data_extended.set_index('Date')[covariates].resample('ME').ffill()
    data_cov_extended = data_cov_extended.astype(float)

    # Separate the historical portion of the exogenous dataset (aligned with our 'ts' index)
    train_exog = data_cov_extended.loc[ts.index]

    # Fill in any missing 2024 data in the training exog with the predicted 2024 values
    for date in train_exog.index:
        if date.year == 2024:
            for cov in covariates:
                if pd.isna(train_exog.loc[date, cov]):
                    train_exog.loc[date, cov] = future_predictions[cov].get('2024', np.nan)

    # Convert everything to float to avoid any dtype issues
    train_exog = train_exog.astype(float)
    future_exog = future_exog.astype(float)

    # =============================================================================
    # 2c) FIT THE SARIMAX MODEL (ARIMA + SEASONAL + EXOGENOUS COVARIATES)
    # =============================================================================

    try:
        # Specific model parameters are chosen: (1,1,0)x(1,0,0,12)
        # The last "12" implies monthly seasonality and the (1,1,0)x(1,0,0) is chosen from
        # all degree-one SARIMA possibilites as this specification outperformed the others
        # in my previous work.
        model = SARIMAX(ts, exog=train_exog, order=(1, 1, 0), seasonal_order=(1, 0, 0, 12))
        results = model.fit(disp=False)
    except Exception as e:
        print(f"Model fitting failed for ZIP {zip_code}: {e}")
        continue

    # =============================================================================
    # 2d) GENERATE FORECASTS AND CONFIDENCE INTERVALS
    # =============================================================================

    try:
        # Forecast out 12 months using the future exogenous values
        forecast_obj = results.get_forecast(steps=forecast_steps, exog=future_exog)
        forecast_log = forecast_obj.predicted_mean      # Forecasts in log-scale
        conf_int_log = forecast_obj.conf_int()          # Confidence intervals in log-scale
    except Exception as e:
        print(f"Forecasting failed for ZIP {zip_code}: {e}")
        continue

    # Convert the log-scale forecasts back to the original scale using exponentiation
    forecast_vals = np.exp(forecast_log)
    conf_int_vals = np.exp(conf_int_log)
    
    # Similarly, get the historical values back on the original scale
    historical_vals = np.exp(ts)

    # =============================================================================
    # 2e) COMPUTE ERROR METRICS (using last 12 months of historical data)
    # =============================================================================

    # We'll treat the first 12 months of forecast as a 'test' set, comparing to actual historical data
    actual = historical_vals[-12:]
    predicted = forecast_vals[:12]

    # Calculate Mean Absolute Error (MAE), Mean Squared Error (MSE), Root MSE
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)

    # =============================================================================
    # 2f) STORE RESULTS AND PREPARE FOR PLOTTING
    # =============================================================================

    # Create a DataFrame holding all forecasted data and their confidence intervals
    forecast_df = pd.DataFrame({
        'ZIP': zip_code,
        'Date': forecast_vals.index,
        'Forecast_ZHVI': forecast_vals.values,
        'Lower_95_CI': conf_int_vals.iloc[:, 0].values,
        'Upper_95_CI': conf_int_vals.iloc[:, 1].values,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    })
    forecast_results.append(forecast_df)

    # Add the historical data trace to the Plotly figure (lighter line for historical)
    fig.add_trace(go.Scatter(
        x=historical_vals.index,
        y=historical_vals,
        mode='lines',
        line=dict(width=1),
        name=f"{zip_code} Historical",
        opacity=0.5
    ))
    
    # Add the forecast data trace to the Plotly figure (bolder line for forecast)
    fig.add_trace(go.Scatter(
        x=forecast_vals.index,
        y=forecast_vals,
        mode='lines',
        line=dict(width=2),
        name=f"{zip_code} Forecast"
    ))

# =============================================================================
# 3) OUTPUT THE RESULTS AND SHOW THE PLOT
# =============================================================================

# Create an output directory if it doesn't already exist
output_dir = "SARIMA_output"
os.makedirs(output_dir, exist_ok=True)

# Path for saving a CSV of the forecast results
forecast_output_path = os.path.join(output_dir, "ZHVI_forecast_Colorado_Springs.csv")

# If we have at least one forecast, concatenate and save them all
if forecast_results:
    full_forecast_df = pd.concat(forecast_results, ignore_index=True)
    full_forecast_df.to_csv(forecast_output_path, index=False)
    print(f"Forecast results saved to {forecast_output_path}")
else:
    print("No forecasts were successfully generated.")

# Make the Plotly figure more informative by adding axis labels and a title
fig.update_layout(
    title="SARIMA Forecasts for Colorado Springs ZIP Codes",
    xaxis_title="Date",
    yaxis_title="ZHVI_sfr",
    showlegend=False
)

# Finally, display the interactive plot
fig.show()
