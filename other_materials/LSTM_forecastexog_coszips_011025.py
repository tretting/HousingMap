# =============================================================================
# This code performs LSTM forecasting of Zillow Home Value Index (ZHVI) for
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
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import plotly.graph_objects as go
import warnings

# Ignore certain warnings for a cleaner output
warnings.filterwarnings("ignore")

# =============================================================================
# 1. LOAD DATA & INITIAL PREPROCESS
# =============================================================================
file_path = "zhvi_allbr_MLdata_010825.csv"
df = pd.read_csv(file_path)

# Some CSVs have an 'Unnamed: 0' column if saved from pandas. We'll remove it.
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# Make sure 'Date' is a proper datetime type and ZIP code (RegionName) is a string
df['Date'] = pd.to_datetime(df['Date'])
df['RegionName'] = df['RegionName'].astype(str)

# Focus only on Colorado Springs data from 2011 onward, and ensure ZHVI is > 0
df = df[(df['City'] == "Colorado Springs") & 
        (df['Date'] >= pd.Timestamp("2011-01-01")) & 
        (df['ZHVI_sfr'] > 0)]

# Sort by ZIP code and then by date
df = df.sort_values(by=['RegionName', 'Date'])

# Convert ZHVI to a log scale, which often stabilizes variance and helps with training
df['log_ZHVI_sfr'] = np.log(df['ZHVI_sfr'])

# Quick look at the range of original and log-transformed ZHVI
print("Log-transformed ZHVI statistics:")
print(df[['ZHVI_sfr', 'log_ZHVI_sfr']].describe())

# Gather all unique ZIP codes and define which covariates we want to consider
zip_codes = df['RegionName'].unique()
covariates = ['LFP_Pop_16', 'Nom_med_inc']

# Prepare an output directory for saving results
output_dir = "LSTM_output"
os.makedirs(output_dir, exist_ok=True)

# Lists to store results and final forecasts
all_forecast_results = []
forecasts_all = {}

# =============================================================================
# Process Each ZIP Code Individually
# =============================================================================
for zip_code in zip_codes:
    print(f"\nProcessing ZIP Code: {zip_code}")
    
    # Filter the big dataframe for just one ZIP code
    df_zip = df[df['RegionName'] == zip_code].copy()
    
    # If less than 24 data points, there's probably insufficient data to train
    if len(df_zip) < 24:
        print(f"Skipping ZIP {zip_code}: Not enough data.")
        continue

    # Make 'Date' the index and sort chronologically
    df_zip.set_index('Date', inplace=True)
    df_zip = df_zip.sort_index()

    # -----------------------------------------------------------------------------
    # Regression-Based Extrapolation and Future Extension for this ZIP code
    # -----------------------------------------------------------------------------
    # We'll take the yearly average of the covariates and assign each year a 
    # numeric 'time' for linear trend estimation
    yearly = df_zip[covariates].resample('Y').mean()
    yearly['time'] = np.arange(len(yearly))

    # We'll store predicted future covariate values in this dictionary
    future_exog_values = {cov: {} for cov in covariates}
    for cov in covariates:
        y = yearly[cov].dropna()
        
        # If there's no or very little data, just take the last known value
        if len(y) < 2:
            val = y.iloc[-1] if not y.empty else np.nan
            future_exog_values[cov]['2024'] = val
            future_exog_values[cov]['2025'] = val
        else:
            # Otherwise, perform a simple linear regression on 'time'
            X = sm.add_constant(yearly.loc[y.index, 'time'])
            model_cov = sm.OLS(y, X).fit()
            # We'll guess future values for the next 2 years (2024, 2025)
            next_times = [len(yearly), len(yearly) + 1]
            preds = model_cov.predict(sm.add_constant(next_times))
            future_exog_values[cov]['2024'] = preds[0]
            future_exog_values[cov]['2025'] = preds[1]

    # Remove any duplicated dates just in case, keeping the first occurrence
    df_zip = df_zip[~df_zip.index.duplicated(keep='first')]
    
    # Identify the last actual date we have data for
    last_actual_date = df_zip.index.max()
    print(f"Last actual date for ZIP {zip_code}: {last_actual_date}")

    # We plan to forecast out until the end of 2025
    end_forecast = pd.Timestamp("2025-12-31")
    
    # Create a monthly DateTimeIndex for the extension window
    month_range = pd.date_range(
        start=last_actual_date + pd.offsets.MonthBegin(1),
        end=end_forecast,
        freq='MS'  # 'MS' means Month Start
    )

    # For each month in that future range, we'll assign predicted covariates
    extension_rows = []
    for d in month_range:
        year_str = str(d.year)
        row = {'Date': d, 'RegionName': zip_code}
        for cov in covariates:
            # If we have a predicted value for this year, use it
            if year_str in future_exog_values[cov]:
                row[cov] = future_exog_values[cov][year_str]
            else:
                # If not specifically predicted, leave it as NaN for now
                row[cov] = np.nan
        extension_rows.append(row)

    # Convert these rows to a DataFrame and set 'Date' as the index
    df_future = pd.DataFrame(extension_rows).set_index('Date')

    # Combine the historical data with future monthly placeholders
    df_combined = pd.concat([df_zip, df_future], sort=True)
    
    # Forward-fill the covariates so that the future months carry the last known
    df_combined[covariates] = df_combined[covariates].ffill()
    df_combined.sort_index(inplace=True)

    # =============================================================================
    # Backtesting: Split data into training and testing sets
    # =============================================================================
    # We'll use the last 12 months of actual data for testing
    test_period = 12  
    if len(df_zip) < (24 + test_period):
        # We need at least 24 months of 'core' data plus 12 for testing
        print(f"Skipping ZIP {zip_code}: Not enough data for backtesting.")
        continue

    # Segment out training data (all but last 12 months) and the testing data
    train_data = df_zip.iloc[:-test_period].copy()
    test_data = df_zip.iloc[-test_period:].copy()

    # =============================================================================
    # Train LSTM on training set
    # =============================================================================

    # Scale the log ZHVI values to [0,1] so the LSTM can handle them more easily
    scaler_train = MinMaxScaler(feature_range=(0, 1))
    train_data['scaled_log_ZHVI'] = scaler_train.fit_transform(train_data[['log_ZHVI_sfr']])
    
    # A helper function to create sequential windows of length 12
    def create_sequences(data, seq_length=12):
        X, y = [], []
        for i in range(len(data) - seq_length):
            # Collect the previous 12 data points as an input
            X.append(data[i:i+seq_length])
            # The next data point is the target
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    # Convert our scaled log ZHVI training data to sequences
    train_seq = train_data['scaled_log_ZHVI'].values.reshape(-1, 1)
    X_train, y_train = create_sequences(train_seq, seq_length=12)
    
    # If no sequences can be created, skip
    if len(X_train) == 0:
        print(f"Skipping ZIP {zip_code}: Not enough sequential training data.")
        continue

    # Build a simple LSTM model with 64 units, plus dropout for generalization
    lstm_model_bt = Sequential([
        LSTM(64, return_sequences=False, input_shape=(12, 1)),
        Dropout(0.2),
        Dense(1)
    ])
    # Compile with MSE loss and the Adam optimizer
    lstm_model_bt.compile(loss="mse", optimizer="adam")
    
    # Fit the model quietly (verbose=0) for 50 epochs
    lstm_model_bt.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

    # =============================================================================
    # Forecast Test Set Period with Monte Carlo Dropout
    # =============================================================================

    # The idea: repeatedly sample the next step with dropout engaged, to get 
    # a distribution of possible outcomes
    
    # We'll start with the last 12 scaled data points from the training set
    last_12_scaled_bt = train_seq[-12:]
    X_last_bt = last_12_scaled_bt.reshape((1, 12, 1))
    
    num_samples = 100  # We'll do 100 Monte Carlo simulations
    test_forecast_steps = len(test_data)
    all_sampled_forecasts_bt = []

    for sample in range(num_samples):
        # Each sample is a new forward simulation
        X_sample_bt = X_last_bt.copy()
        sample_preds_bt = []
        for step in range(test_forecast_steps):
            # Model is called with training=True to ensure dropout is active
            next_pred_scaled_bt = lstm_model_bt(X_sample_bt, training=True).numpy()[0, 0]
            sample_preds_bt.append(next_pred_scaled_bt)
            # Update the sequence by dropping the oldest step and adding the new prediction
            X_sample_bt = np.append(X_sample_bt[:, 1:, :], [[[next_pred_scaled_bt]]], axis=1)
        all_sampled_forecasts_bt.append(sample_preds_bt)

    # Convert to a NumPy array for easy manipulation
    all_sampled_forecasts_bt = np.array(all_sampled_forecasts_bt)
    
    # Inverse-transform from scaled log ZHVI back to log ZHVI
    all_sampled_forecasts_log_bt = scaler_train.inverse_transform(all_sampled_forecasts_bt.T).T
    
    # Convert from log scale to the original ZHVI scale
    all_sampled_forecasts_ZHVI_bt = np.exp(all_sampled_forecasts_log_bt)

    # Take the mean across all Monte Carlo samples as our final forecast
    forecast_mean_test = np.mean(all_sampled_forecasts_ZHVI_bt, axis=0)

    # Compare to the actual test data
    actual_test = test_data['ZHVI_sfr'].values

    # Calculate error metrics
    mae = mean_absolute_error(actual_test, forecast_mean_test)
    mse = mean_squared_error(actual_test, forecast_mean_test)
    rmse = np.sqrt(mse)

    print(f"Backtesting metrics for ZIP {zip_code}: MAE={mae:.2f}, MSE={mse:.2f}, RMSE={rmse:.2f}")

    # =============================================================================
    # LSTM Forecasting for this ZIP Code
    # =============================================================================

    # We'll now build the final LSTM model using the entire historical range (up to last_actual_date).
    historical_data = df_combined[df_combined.index <= last_actual_date].copy()
    
    # Still need at least 24 months of data to train meaningfully
    if len(historical_data) < 24:
        print(f"Skipping ZIP {zip_code}: Not enough historical training data.")
        continue

    # Scale the log ZHVI in the historical data
    scaler = MinMaxScaler(feature_range=(0, 1))
    historical_data['scaled_log_ZHVI'] = scaler.fit_transform(historical_data[['log_ZHVI_sfr']])

    # Reuse the same sequence creation function
    def create_sequences(data, seq_length=12):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    train_seq = historical_data['scaled_log_ZHVI'].values.reshape(-1, 1)
    X_data, y_data = create_sequences(train_seq, seq_length=12)

    # If no valid sequences, skip
    if len(X_data) == 0:
        print(f"Skipping ZIP {zip_code}: Not enough sequential data.")
        continue

    # Build a new LSTM model with the same architecture
    lstm_model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(12, 1)),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(loss="mse", optimizer="adam")
    lstm_model.fit(X_data, y_data, epochs=50, batch_size=8, verbose=0)

    # =============================================================================
    # 8. FORECAST NEXT 12 MONTHS WITH MONTE CARLO DROPOUT
    # =============================================================================

    # Again, we'll start with the last 12 scaled points
    last_12_scaled = train_seq[-12:]
    X_last = last_12_scaled.reshape((1, 12, 1))

    num_samples = 100
    all_sampled_forecasts = []

    # We'll use a separate scaler (scl) to invert the predictions from scaled log to log
    scl = MinMaxScaler(feature_range=(0, 1))
    scl.fit(historical_data[['log_ZHVI_sfr']])

    # Perform Monte Carlo sampling to get a distribution of next 12 months
    for sample in range(num_samples):
        X_sample = X_last.copy()
        sample_preds = []
        for step in range(12):
            # training=True => dropout remains active
            next_pred_scaled = lstm_model(X_sample, training=True).numpy()[0, 0]
            sample_preds.append(next_pred_scaled)
            X_sample = np.append(X_sample[:, 1:, :], [[[next_pred_scaled]]], axis=1)
        all_sampled_forecasts.append(sample_preds)

    all_sampled_forecasts = np.array(all_sampled_forecasts)

    # Inverse scaling from scaled log(ZHVI) to log(ZHVI)
    all_sampled_forecasts_log = scl.inverse_transform(all_sampled_forecasts.T).T
    
    # Convert log(ZHVI) back to the original ZHVI scale
    all_sampled_forecasts_ZHVI = np.exp(all_sampled_forecasts_log)

    # Compute mean, and 2.5 and 97.5 percentiles for a 95% confidence interval
    forecast_mean = np.mean(all_sampled_forecasts_ZHVI, axis=0)
    forecast_lower = np.percentile(all_sampled_forecasts_ZHVI, 2.5, axis=0)
    forecast_upper = np.percentile(all_sampled_forecasts_ZHVI, 97.5, axis=0)

    # =============================================================================
    # 9. RECORD FORECAST RESULTS WITH CONFIDENCE INTERVALS
    # =============================================================================

    # Create a date range for the next 12 months following the last actual date
    forecast_dates = pd.date_range(
        start=last_actual_date + pd.offsets.MonthBegin(1),
        periods=12, freq='M'
    )

    # Compile a DataFrame holding the forecast values and the performance metrics
    forecast_df = pd.DataFrame({
        "ZIP": zip_code,
        "Date": forecast_dates,
        "Forecast_ZHVI": forecast_mean,
        "Lower_95_CI": forecast_lower,
        "Upper_95_CI": forecast_upper,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })

    all_forecast_results.append(forecast_df)
    forecasts_all[zip_code] = (forecast_dates, forecast_mean, forecast_lower, forecast_upper)

# =============================================================================
# 10. SAVE RESULTS & PLOT OVERLAY
# =============================================================================
if all_forecast_results:
    combined_forecast_df = pd.concat(all_forecast_results, ignore_index=True)
    combined_forecast_df.to_csv(
        os.path.join(output_dir, "LSTM_forecast_Colorado_Springs.csv"), 
        index=False
    )
    print("Forecast results saved!")

# We'll generate a simple overlay plot of the different ZIP codes' forecasts
fig = go.Figure()
for zip_code, (f_dates, f_mean, f_lower, f_upper) in forecasts_all.items():
    fig.add_trace(go.Scatter(
        x=f_dates,
        y=f_mean,
        mode="lines",
        name=f"ZIP {zip_code}"
    ))

fig.update_layout(
    title="LSTM Forecasts for Colorado Springs ZIP Codes",
    xaxis_title="Date",
    yaxis_title="ZHVI",
    showlegend=True
)

# Save the figure as a PNG and show it interactively
fig.write_image(os.path.join(output_dir, "LSTM_forecast_plot.png"))
fig.show()