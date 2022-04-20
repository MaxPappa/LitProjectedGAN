from dataclasses import dataclass

@dataclass
class configParams:
    batch_size: int = 64
    epochs: int = 250
    gen_lr: float = 0.0002
    discr_lr: float = 0.0002
    beta1: float = 0.0
    beta2: float = 0.99
    latent_dim: int = 128
    diff_aug: bool = True
    checkpoint_path: str = "./checkpoint"
    checkpoint_efficient_net: str = "efficientnet_lite1.pth"
    log_every: int = 5
    dataset_path: str = "./data"
    image_size: int = 256
    num_epoch_checkpoint: int = 100