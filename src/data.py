import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
import xarray as xr
import numpy as np
import os
from pathlib import Path

class ClimateDataset(Dataset):
    def __init__(self, xr_slice, mu, sigma):
        """
        xr_slice: xarray Dataset (portion of the full data)
        mu: mean values for normalization (xarray)
        sigma: std values for normalization (xarray)
        """
        self.ds = xr_slice
        self.mu = mu
        self.sigma = sigma
        self.n_samples = len(self.ds.time) - 1

    def __len__(self):
        return self.n_samples

    def _process_sample(self, sample_xr):
        """
        Ensures the output is always (2, 32, 64) mapping to (Channel, Lat, Lon).
        """
        # 1. Normalize
        norm_ds = (sample_xr - self.mu) / self.sigma
        
        # 2. Explicitly transpose to (lat, lon) to avoid the 90-degree rotation bug
        # This ensures row index = latitude and column index = longitude
        t2m = norm_ds['2m_temperature'].transpose('latitude', 'longitude').values
        sp  = norm_ds['surface_pressure'].transpose('latitude', 'longitude').values
        
        # 3. Shape Guard: Catch dimension issues before they hit the model
        if t2m.shape[0] > t2m.shape[1]:
            raise ValueError(f"Expected shape (32, 64), but got {t2m.shape}. Check your netCDF dimensions.")

        # 4. Stack into [C, H, W] -> [2, 32, 64]
        return torch.from_numpy(np.stack([t2m, sp])).float()

    def __getitem__(self, idx):
        raw_x = self.ds.isel(time=idx)
        raw_y = self.ds.isel(time=idx + 1)
        
        x_tensor = self._process_sample(raw_x)
        y_tensor = self._process_sample(raw_y)
        
        return x_tensor, y_tensor

class ClimateDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=16, split_date="2021-01-01"):
        super().__init__()
        self.data_path = Path(data_dir) / "era5_real.nc"
        self.batch_size = batch_size
        self.split_date = split_date
        self.mu = None
        self.sigma = None

    def prepare_data(self):
        # This only runs once. We compute stats on the training split only.
        ds = xr.open_dataset(self.data_path)
        train_slice = ds.sel(time=slice(None, self.split_date))
        
        print(f"--- Computing Stats on training period (ending {self.split_date}) ---")
        # We use 'latitude' and 'longitude' specifically to match ERA5 naming
        self.mu = train_slice.mean(dim=['time', 'latitude', 'longitude']).compute()
        self.sigma = train_slice.std(dim=['time', 'latitude', 'longitude']).compute()
        
        # Prevent division by zero
        self.sigma = xr.where(self.sigma == 0, 1.0, self.sigma)

    def setup(self, stage=None):
        full_ds = xr.open_dataset(self.data_path)

        if stage == "fit" or stage is None:
            train_ds_raw = full_ds.sel(time=slice(None, self.split_date))
            val_ds_raw = full_ds.sel(time=slice(self.split_date, None))

            self.train_ds = ClimateDataset(train_ds_raw, self.mu, self.sigma)
            self.val_ds = ClimateDataset(val_ds_raw, self.mu, self.sigma)
            
            # Final sanity check on a single sample
            sample_x, _ = self.train_ds[0]
            print(f"--- Data Setup Complete. Tensor Shape: {sample_x.shape} (Expect [2, 32, 64]) ---")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )