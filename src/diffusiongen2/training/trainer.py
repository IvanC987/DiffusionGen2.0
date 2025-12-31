import os
import time
import json
from pathlib import Path
from logging import Logger
import torch
import wandb
from PIL import Image
from diffusers import AutoencoderKL
from wandb import Run
from hydra.core.hydra_config import HydraConfig

from diffusiongen2.data.dataset_loader import DatasetLoader
from diffusiongen2.configs import ProjectConfig
from diffusiongen2.utils.paths import PROJECT_ROOT
from diffusiongen2.models.diffusion import DiffusionModel


@torch.no_grad()
def generate_and_log_samples(model: DiffusionModel,
                             vae: AutoencoderKL,
                             eval_dict: dict,
                             eval_output_dir: Path,
                             latent_dim: int,
                             epoch: int,
                             cfg_scales: tuple,
                             device: str):
    start = time.time()

    os.makedirs(eval_output_dir / str(epoch), exist_ok=True)

    prompts: list = eval_dict["prompts"]
    eval_prompt_embd: torch.Tensor = eval_dict["text_embd"]  # Tensor of shape (B, 77, 512)
    eval_null_embd: torch.Tensor = eval_dict["null_embd"]  # Tensor of shape (1, 77, 512)

    # Test out different cfg across range like [2, 4, 6, 8] for each image, total of B * len(cfg_scales) images
    len_cfg_scales = len(cfg_scales)
    cfg_scales = torch.tensor(cfg_scales, dtype=torch.float32, device=device)

    # Turn from (1, 77, 512) to (len_cfg_scales, 77, 512)
    negative_prompts = eval_null_embd.repeat(len_cfg_scales, 1, 1)

    wandb_images = []
    for i, single_prompt_embd in enumerate(eval_prompt_embd):
        # Same shape as 'negative_prompts' tensor
        single_prompt_embd = single_prompt_embd.unsqueeze(0).repeat_interleave(len_cfg_scales, dim=0)

        with torch.autocast(device_type=device, dtype=torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32):
            # 'latents' should be of shape (len_cfg_scales, 4, H//8, W//8)
            latents = model.generate(prompt=single_prompt_embd, negative_prompts=negative_prompts, latent_dim=latent_dim, cfg_scale=cfg_scales)

        # Now decode and save the samples, shape (len_cfg_scales, 3, H, W)
        generated_images = vae.decode(latents / 0.18215).sample

        # Rescale images to [0, 1] range
        generated_images = (generated_images.clamp(-1, 1) + 1) / 2.0

        for j, img_tensor in enumerate(generated_images):  # Iterate over batch dim
            # Permute to (H, W, 3) and convert to 0-255 uint8
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_uint8 = (img_np * 255).round().astype("uint8")

            # Create PIL Image and save
            img_final = Image.fromarray(img_uint8)
            img_final.save(eval_output_dir / str(epoch) / f"{i*len_cfg_scales + j:02d}.png")
            wandb_images.append(wandb.Image(img_final, caption=f"{prompts[i]}\ncfg={cfg_scales[j].item()}"))

    return eval_prompt_embd.shape[0] * len_cfg_scales, int(time.time() - start), wandb_images


# TODO: Later on, add EMA for weight updates
def train(model: DiffusionModel,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          scheduler: torch.optim.lr_scheduler.LRScheduler,
          vae: AutoencoderKL,
          dataset_loader: DatasetLoader,
          logger: Logger,
          run: Run,
          config: ProjectConfig,
          T: int,
          device: str):

    training_config = config.training
    optim_config = config.optim
    eval_config = config.eval

    hydra_out_dir = Path(HydraConfig.get().runtime.output_dir)

    # Set up checkpointing directory
    ckpt_dir = hydra_out_dir / "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Compile model as needed
    if training_config.compile_model:
        logger.info("Compiling model...")
        raw_model = model
        model = torch.compile(model)
    else:
        raw_model = model

    # Eval Prompts
    eval_prompts_path = PROJECT_ROOT / eval_config.prompts_path
    eval_output_dir = hydra_out_dir / "samples"
    os.makedirs(eval_output_dir, exist_ok=True)
    eval_dict = torch.load(eval_prompts_path, map_location=device)
    with open(eval_output_dir / "prompts.txt", "w") as f:
        for prompt in eval_dict["prompts"]:
            f.write(f"{prompt}\n")

    # Initialize trackers
    train_losses, train_losses_ema, val_losses = [], [], []
    ema_train_loss = 0
    global_train_step = 1

    mini_steps = max(1, training_config.effective_batch_size // training_config.batch_size)
    for epoch in range(training_config.epochs):
        logger.info(f"Currently at Epoch {epoch+1}")

        # Train
        # -------------------------------
        local_train_step = 0
        start = time.time()
        while dataset_loader.train_epoch == epoch:
            optimizer.zero_grad(set_to_none=True)

            loss_accum = 0
            for mini_step in range(mini_steps):
                latents, text_embd = dataset_loader.get_batch(train=True)
                timesteps = torch.randint(1, T, (latents.shape[0],)).to(device)
                latents, text_embd = latents.to(device), text_embd.to(device)

                with torch.autocast(device_type=device, dtype=torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32):
                    pred, target = model(latents, text_embd, timesteps)
                    loss = criterion(pred, target)  # Looking back in my v1.0 project, it seems like I decoded the latent noise to compute diff...*FacePalm*

                loss /= mini_steps
                loss_accum += loss.item()
                loss.backward()

            train_losses.append([global_train_step, loss_accum])
            ema_train_loss = (1 - training_config.ema_beta) * loss_accum + training_config.ema_beta * ema_train_loss

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optim_config.grad_clip)  # Prevents unstable learning
            optimizer.step()

            if local_train_step % eval_config.log_every_n_steps == 0:
                elapsed = time.time() - start
                start = time.time()

                # Bias correction for initial steps
                train_loss_ema_bc = ema_train_loss / (1 - training_config.ema_beta ** global_train_step)
                train_losses_ema.append([global_train_step, train_loss_ema_bc])

                logger.info(f"Training Loss: {train_loss_ema_bc:.4f}   |   "  
                            f"Learning Rate: {optimizer.param_groups[0]['lr']:.5f}   |   "
                            f"Norm: {norm.item():.4f}   |   "
                            f"Time: {int(elapsed)}s")

                run.log({
                    "train/loss_ema": train_loss_ema_bc,
                    "train/loss": loss_accum,
                    "optim/lr": optimizer.param_groups[0]["lr"],
                    "optim/grad_norm": norm.item(),
                    "perf/train_log_interval_time": elapsed,
                    "epoch": epoch,
                }, step=global_train_step)

            local_train_step += 1
            global_train_step += 1
        # -------------------------------


        # Validation
        # -------------------------------
        if epoch % eval_config.val_every_n_epochs == 0 or epoch == training_config.epochs - 1:
            model.eval()

            local_val_loss = []
            val_timer = time.time()
            with torch.no_grad():
                while dataset_loader.val_epoch == epoch:
                    latents, text_embd = dataset_loader.get_batch(train=False)
                    timesteps = torch.randint(1, T, (latents.shape[0],)).to(device)
                    latents, text_embd = latents.to(device), text_embd.to(device)

                    with torch.autocast(device_type=device, dtype=torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32):
                        pred, target = model(latents, text_embd, timesteps)
                        loss = criterion(pred, target)

                    local_val_loss.append(loss.item())

            elapsed = time.time() - val_timer
            avg_val_loss = sum(local_val_loss)/len(local_val_loss)
            logger.info(f"Validation Loss: {avg_val_loss:.4f}   |   Time: {int(elapsed)}s")
            run.log({
                "val/loss": avg_val_loss,
                "val/time": elapsed,
                "epoch": epoch,
            }, step=global_train_step)
            val_losses.append([global_train_step, avg_val_loss])

            model.train()
        # -------------------------------

        # Generate sample images
        # -------------------------------
        if epoch % eval_config.sample_every_n_epochs == 0 or epoch == training_config.epochs - 1:
            # A bit of a hacky solution, but don't want to set img_dim in configs. Might forget to update when testing out various resolutions. Better to get from DS Loader
            latent_dim = dataset_loader.train_latent.shape[2]

            samples_output = generate_and_log_samples(model=model, vae=vae, eval_dict=eval_dict,
                                                      eval_output_dir=eval_output_dir, latent_dim=latent_dim,
                                                      epoch=epoch, cfg_scales=eval_config.cfg_scales, device=device
                                                      )

            num_samples_generated, sample_generation_time, wandb_images = samples_output
            logger.info(f"Generated {num_samples_generated} images. Took {sample_generation_time}s")
            run.log({
                "samples": wandb_images,
                "epoch": epoch,
            }, step=global_train_step)
        # -------------------------------


        # Step scheduler and save checkpoint as needed
        scheduler.step()

        # Save/Update the training losses
        metrics_json = {
            "train_losses": train_losses,
            "train_losses_ema": train_losses_ema,
            "val_losses": val_losses,
        }
        with open(hydra_out_dir / "metrics.json", "w") as f:
            json.dump(metrics_json, f)

        if epoch % training_config.ckpt_interval == 0 or epoch == training_config.epochs - 1:
            filename = f"epoch_{epoch:04d}_tl_{ema_train_loss:.4f}.pt"
            ckpt_path = ckpt_dir / filename

            ckpt_dict = {
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "epoch": epoch,
                "train_losses": train_losses,
                "val_losses": val_losses,
            }

            torch.save(ckpt_dict, ckpt_path)
