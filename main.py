import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from litProjectedGAN import litProjectedGAN
from dataset import load_data
import wandb
import config


class EpochModelCheckpoint(pl.Callback):
    def __init__(self, num_epoch: int):
        self.num_epoch = num_epoch

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if (trainer.current_epoch % self.num_epoch) == 0 and trainer.current_epoch > 0:
            trainer.save_checkpoint(f"./checkpoint/epoch={trainer.current_epoch}_LitProjectedGAN.pth")
            wandb.save(f"./checkpoint/epoch={trainer.current_epoch}_LitProjectedGAN.pth")



if __name__ == '__main__':
    cfg = config.configParams()
    data = load_data(cfg.dataset_path, cfg.batch_size)
    model = litProjectedGAN(cfg)
    wandb_logger = WandbLogger(project="LitProjectedGAN", name="art-painting")
    trainer = pl.Trainer(
        devices=1, accelerator="gpu",
        log_every_n_steps=cfg.log_every, max_epochs=200,
        num_sanity_val_steps=0, logger=wandb_logger, precision=16,
        callbacks=[EpochModelCheckpoint(num_epoch=50)], checkpoint_callback=False
    )
    # checkpoint_callback=False avoid saving per-epoch temporary checkpoints (saving a lot of time)
    trainer.fit(model, data)
    wandb.save(f"./checkpoint/*")
    wandb.finish()