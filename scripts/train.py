import logging
import random

import hydra
import numpy as np
import wandb
import torch
from pathlib import Path
from diffusers import AutoencoderKL
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffusiongen2.configs import ProjectConfig
from diffusiongen2.models.diffusion import DiffusionModel
from diffusiongen2.data.dataset_loader import DatasetLoader
from diffusiongen2.training.trainer import train

cs = ConfigStore.instance()
cs.store(name="config", node=ProjectConfig)


def set_seed(seed: int, device: str):
    # Set Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in device:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_logger():
    # Hydra sets it up now
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s [%(levelname)s] %(message)s",
    #     handlers=[
    #         logging.StreamHandler(sys.stdout)  # Prints to console
    #     ]
    # )

    logger = logging.getLogger(__name__)
    return logger


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(config: DictConfig):
    logger = get_logger()

    logger.info("==== CONFIG BEGIN ====")
    for line in OmegaConf.to_yaml(config).splitlines():
        logger.info(line)
    logger.info("==== CONFIG END ====")

    hydra_out_dir = Path(HydraConfig.get().runtime.output_dir)
    run = wandb.init(
        project="DiffusionGen2.0",
        name=hydra_out_dir.name,
        dir=str(hydra_out_dir),
        config=OmegaConf.to_container(config, resolve=True),
    )

    config: ProjectConfig = OmegaConf.to_object(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Enable TF32, set seed, and get logger
    torch.set_float32_matmul_precision("high")
    set_seed(config.training.seed, device)

    dataset_loader = DatasetLoader(
        shard_directory=config.data.shard_dir,
        batch_size=config.training.batch_size,
        val_split=config.data.val_split,
        embd_dropout=config.training.cfg_dropout
    )
    dataset_loader.log_info(logger=logger, effective_batch_size=config.training.effective_batch_size)

    model = DiffusionModel(diffusion_config=config.diffusion, unet_config=config.unet).to(device)
    optimizer = AdamW(model.parameters(), lr=config.optim.max_lr, weight_decay=config.optim.weight_decay)
    criterion = torch.nn.MSELoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.epochs, eta_min=config.optim.min_lr)

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(config.data.vae_model_id).to(device)
    vae.requires_grad_(False)
    vae.eval()

    logger.info(f"There are {sum(p.numel() for p in model.parameters()):,} parameters in DiffusionModel")
    logger.info(f"There are {sum(p.numel() for p in vae.parameters()):_} parameters in VAE")


    train(model=model,
          optimizer=optimizer,
          criterion=criterion,
          scheduler=scheduler,
          vae=vae,
          dataset_loader=dataset_loader,
          logger=logger,
          run=run,
          config=config,
          T=config.diffusion.T,
          device=device)

    run.finish()


if __name__ == "__main__":
    main()
