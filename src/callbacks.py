import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import os
import glob
import numpy as np
import geopandas as gpd
from PIL import Image

class VisualizationCallback(pl.Callback):
    def __init__(self, snapshot_interval=10):
        super().__init__()
        self.snapshot_interval = snapshot_interval
        self.viz_path = None
        self.snapshots_path = None
        self.val_batch = None
        
        # Consistent scaling parameters
        self.fixed_vmin = None
        self.fixed_vmax = None
        
        try:
            world_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
            self.world = gpd.read_file(world_url)
        except Exception as e:
            print(f"Warning: Could not load world map shapes: {e}")
            self.world = None

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.global_rank != 0:
            return

        current_epoch = trainer.current_epoch
        # Ensure the last epoch is ALWAYS captured
        is_last_epoch = (current_epoch + 1) == trainer.max_epochs
        should_viz = (current_epoch % self.snapshot_interval == 0) or is_last_epoch

        if not should_viz:
            return

        if self.viz_path is None:
            root_dir = trainer.log_dir if trainer.log_dir else "lightning_logs/version_0"
            self.viz_path = os.path.join(root_dir, 'visualizers')
            self.snapshots_path = os.path.join(self.viz_path, 'snapshots')
            os.makedirs(self.snapshots_path, exist_ok=True)

        if self.val_batch is None:
            val_loader = trainer.datamodule.val_dataloader()
            self.val_batch = next(iter(val_loader))

        x_in, y_true = self.val_batch
        x_in, y_true = x_in.to(pl_module.device), y_true.to(pl_module.device)
        
        pl_module.eval()
        with torch.no_grad():
            pred_norm = pl_module(x_in)
        pl_module.train()

        mu_t2m = trainer.datamodule.mu['2m_temperature'].values
        sigma_t2m = trainer.datamodule.sigma['2m_temperature'].values
        
        def process_map(tensor_chan):
            arr = tensor_chan.cpu().numpy()
            return (arr * sigma_t2m) + mu_t2m

        truth_map = process_map(y_true[0, 0])
        pred_map = process_map(pred_norm[0, 0])
        error_map = pred_map - truth_map

        # SET CONSISTENT SCALE (Locked to the first truth map encountered)
        if self.fixed_vmin is None:
            self.fixed_vmin = truth_map.min()
            self.fixed_vmax = truth_map.max()

        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        fig.suptitle(f"Epoch: {current_epoch:03d} | 2m Temperature (K)", fontsize=20)

        extent = [-180, 180, -90, 90]
        plot_data = [truth_map, pred_map, error_map]
        titles = ["Ground Truth (t+1)", "Prediction (t+1)", "Error (Pred - Truth)"]
        cmaps = ['RdYlBu_r', 'RdYlBu_r', 'seismic']

        for i, ax in enumerate(axes):
            if self.world is not None:
                self.world.boundary.plot(ax=ax, color='black', linewidth=0.8, zorder=2)
            
            if i < 2:
                # Use the fixed scale for Truth and Prediction
                vmin, vmax = self.fixed_vmin, self.fixed_vmax
            else:
                # Error scale is always centered at 0
                # We cap the error scale so tiny fluctuations don't look massive
                vmin, vmax = -10, 10 

            im = ax.imshow(plot_data[i], extent=extent, cmap=cmaps[i], 
                           vmin=vmin, vmax=vmax, aspect='equal', origin='upper', zorder=1)
            
            ax.set_title(titles[i], fontsize=14)
            plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        filename = f"epoch_{current_epoch:04d}.png"
        save_path = os.path.join(self.snapshots_path, filename)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.global_rank != 0 or not self.snapshots_path:
            return

        gif_path = os.path.join(self.viz_path, "training_evolution.gif")
        img_files = sorted(glob.glob(os.path.join(self.snapshots_path, "*.png")))
        
        if img_files:
            frames = [Image.open(f) for f in img_files]
            
            # DYNAMIC DURATION: 
            # We want the GIF to last roughly 5-10 seconds total.
            # Total duration (ms) / number of frames
            # Max 1000ms (slow), Min 100ms (fast)
            duration = max(100, min(1000, 5000 // len(frames)))
            
            # The last frame (the final model state) stays visible longer (2 seconds)
            durations = [duration] * (len(frames) - 1) + [2000]
            
            frames[0].save(
                gif_path, 
                save_all=True, 
                append_images=frames[1:], 
                duration=durations, 
                loop=0
            )
            print(f"Visualizer: GIF saved to {gif_path} with {duration}ms frame delay.")