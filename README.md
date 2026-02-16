# CMCC PyTorch Lightning Weather Forecasting

## Introduction

This project implements a deep learning model for global weather forecasting using **PyTorch Lightning**. It utilizes a subset of the ERA5 reanalysis dataset to predict future atmospheric states (Temperature and Surface Pressure) on a coarse global grid.

The system is designed to demonstrate a complete machine learning pipeline for climate data, including efficient data loading with Xarray, a residual Convolutional Neural Network (CNN) with periodic boundary conditions, and automated visualization of training progress.

## How to use

In order to use it, first create a virtual environment and install from `requirements.txt`, then download the dataset by launching `download_era5_data.py`:

```bash
pip install -r requirements.txt
python download_era5_data.py
```

Then set the project parameters in `configs/base_config.yaml` or create a new config and run:

```bash
python main.py fit --config configs/base_config.yaml
```

You will find the results in `lightning_logs/example_experiment/{experiment_start_timestamp}` in which you will also find a gif visualization of the training.

## Dataset

The project uses **ERA5 reanalysis data** sourced from the WeatherBench 2 project on Google Cloud.

- **Source**: `gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr`
- **Variables**:
  - `2m_temperature` (Air temperature at 2 meters)
  - `surface_pressure`
- **Resolution**: 32 latitude x 64 longitude (approx. 5.625°).
- **Time Range**: The download script retrieves data from 2015 to 2022 to save disk space.

## Data Module

The data handling logic is encapsulated in `src/data.py`:

- **`ClimateDataModule`**: Manages the training and validation splits based on a specific date (`2021-01-01`). It computes global mean and standard deviation statistics on the training set to perform Z-score normalization.
- **`ClimateDataset`**: Wraps the Xarray data. It yields pairs of `(current_time, next_time)` tensors. It ensures the data is shaped as `[Channels, Latitude, Longitude]` and handles the conversion from NetCDF to PyTorch tensors.

## Model

The core logic is located in `src/models.py`:

- **`WeatherModel`**: A simple CNN that employs **Residual Learning**. Instead of predicting the absolute weather values for the next step, it predicts the *change* (delta) from the current step. This generally leads to more stable training for time-series physical systems.
- **Periodic Padding**: The model includes a custom `PeriodicPadding2d` layer. Since the Earth is a sphere, the left edge of the map (Longitude -180) connects to the right edge (Longitude +180). This layer pads the input circularly along the longitude dimension to ensure the convolution operations respect this topology.

## Callbacks

Visualization and monitoring are handled in `src/callbacks.py`:

- **`VisualizationCallback`**:
  - Runs at the end of validation epochs.
  - Generates a 3-panel plot comparing:
    1. Ground Truth (Next Step)
    2. Model Prediction
    3. Error Map (Prediction - Truth)
  - Uses `geopandas` to overlay world coastlines for better context.
  - **GIF Generation**: At the end of training, it stitches all saved epoch snapshots into a `training_evolution.gif`, allowing you to watch the model learn the weather patterns over time.