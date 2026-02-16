import torch
import datetime
from pytorch_lightning.cli import LightningCLI
from src.models import WeatherModel
from src.data import ClimateDataModule

# Enable Tensor Core optimization for your 4070 SUPER
torch.set_float32_matmul_precision('high')

class TimestampedCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # We only set the default logger here to ensure a unique timestamped folder
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M")
        parser.set_defaults({
            "trainer.logger": {
                "class_path": "pytorch_lightning.loggers.TensorBoardLogger",
                "init_args": {
                    "save_dir": "lightning_logs",
                    "name": "example_experiment",
                    "version": timestamp
                }
            }
        })

def cli_main():
    TimestampedCLI(
        model_class=WeatherModel,
        datamodule_class=ClimateDataModule,
        save_config_kwargs={"overwrite": True},
        # This allows you to easily switch models/data from the CLI if you make new ones later
        subclass_mode_model=False, 
        subclass_mode_data=False
    )

if __name__ == "__main__":
    cli_main()