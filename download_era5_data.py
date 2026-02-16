# download_data.py
import xarray as xr
import zarr
import gcsfs

def download_subset():
    print("--- Connecting to WeatherBench 2 (GCP) ---")
    path = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr"
    
    # Open the remote Zarr dataset
    ds = xr.open_zarr(path)
    
    # Select variables and a recent time slice to save disk space
    # ERA5 in WB2 uses '2m_temperature' and 'surface_pressure'
    subset = ds[['2m_temperature', 'surface_pressure']].sel(time=slice("2015", "2022"))
    
    print("--- Downloading and saving to ./data/era5_real.nc ---")
    # This will take a few minutes depending on your connection
    subset.to_netcdf("./data/era5_real.nc")
    print("Done!")

if __name__ == "__main__":
    download_subset()