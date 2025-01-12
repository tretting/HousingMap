# Colorado Springs Housing Map

This repository hosts a Flask-based web application for visualizing housing-related data in Colorado Springs. The app includes the ability to plot annual, monthly, and forecasted home value metrics on a ZIP code heatmap.

## Features

- **Annual Metrics:** Socioeconomic data from various sources (e.g., Census).
- **Monthly Metrics:** Zillow Housing Value Index (ZHVI) data at ZIP-level, aggregated and cleaned.
- **Forecast Metrics:** SARIMA and LSTM models predicting ZHVI values for upcoming months.
- **Lot Coverage:** Custom metric showing the percentage of a lot covered by built footprint.

## Key info

This repository includes everything you need to recreate the app here: https://www.andyretting.org/portfolio/coshousingtool, with the exception of the TIGER zipcode shapefile data, which you must download. It is too large to include here. The code to process the data is included, but you must ensure your directory structure and filenames are corretly specified. 


