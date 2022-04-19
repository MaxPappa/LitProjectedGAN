from dataclasses import dataclass

@dataclass
class configParams:
    batch_size: int = 32
    epochs: int = 200
    gen_lr: float = 0.0025
    discr_lr: float = 0.002
    beta1: float = 0.5
    beta2: float = 0.999
    latent_dim: int = 64
    diff_aug: bool = True
    checkpoint_path: str = "./checkpoint"
    checkpoint_efficient_net: str = "efficientnet_lite1.pth"
    log_every: int = 5
    dataset_path: str = "./data"
    image_size: int = 256
    num_epoch_checkpoint: int = 50