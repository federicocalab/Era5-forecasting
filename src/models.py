import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class PeriodicPadding2d(nn.Module):
    def __init__(self, pad_width):
        super().__init__()
        self.pad_width = pad_width

    def forward(self, x):
        # x shape: [Batch, Chan, Lat, Lon] -> [B, C, 32, 64]
        
        # SAFETY CHECK: 
        # In a global grid, Lon (width) is usually 2x the Lat (height).
        # We ensure we are padding the LARGER dimension (64).
        if x.shape[-1] < x.shape[-2]:
            raise ValueError(
                f"Orientation Error: Received shape {list(x.shape)}. "
                f"The last dimension (Longitude) should be the widest for circular padding."
            )

        # F.pad format: (padding_left, padding_right, padding_top, padding_bottom)
        # We pad ONLY left/right (longitude) circularly.
        return F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode='circular')

class WeatherModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, hidden_dim: int = 64):
        super().__init__()
        self.save_hyperparameters()
        
        self.pad = PeriodicPadding2d(1)
        
        # Conv2d padding=(1, 0)
        # 1 pixel for Top/Bottom (Latitude) - standard zero-padding
        # 0 pixels for Left/Right - because self.pad already handled it
        self.encoder = nn.Sequential(
            self.pad, 
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=(1, 0)), 
            nn.ReLU(),
            
            self.pad,
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=(1, 0)),
            nn.ReLU(),
            
            self.pad,
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=(1, 0))
        )
        
    def forward(self, x):
        # RESIDUAL LEARNING: Predict the delta (change) in weather
        return x + self.encoder(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        
        # persistence_loss is our 'baseline' (predicting no change at all)
        persistence_loss = F.mse_loss(x, y)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("persistence_loss", persistence_loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)