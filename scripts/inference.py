import yaml
import torch
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL

from diffusiongen2.models.diffusion import DiffusionModel
from diffusiongen2.configs import ProjectConfig


def load_inference_config(inf_config_path: str) -> dict:
    with open(inf_config_path, "r") as f:
        return yaml.safe_load(f)


def load_vae(model_id: str, device: str) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(model_id).to(device)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def load_model_from_checkpoint(checkpoint_paths: str, use_ema: bool, device: str) -> DiffusionModel:
    # Was planning to use 'torch.serialization.add_safe_globals' but realized I would need to add that for each config file
    # Instead just use weights_only=False would fix most.
    # This is purely for the configs dataclasses in 'src/diffusiongen2/configs.py' classes
    saved_checkpoint = torch.load(checkpoint_paths, map_location=device, weights_only=False)
    training_configs: ProjectConfig = saved_checkpoint["config"]

    model = DiffusionModel(diffusion_config=training_configs.diffusion, unet_config=training_configs.unet).to(device)
    model.requires_grad_(False)

    if use_ema:
        model.load_state_dict(saved_checkpoint["ema_state_dict"])
    else:
        model.load_state_dict(saved_checkpoint["model_state_dict"])

    del saved_checkpoint  # Can be quite large, del after loading
    model.eval()
    return model


@torch.inference_mode()
def generate_latent(model: DiffusionModel,
                    vae: AutoencoderKL,
                    prompt: torch.Tensor,
                    negative_prompt: torch.Tensor,
                    cfg: float,
                    T: int,
                    batch_size: int,
                    latent_dim: int,
                    use_ddpm: bool,
                    num_steps: int,
                    seed: int | None,
                    device: str,
                    preview_interval: int,
                    preview_callback=None,
                    progress_callback=None) -> list[Image]:

    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

    # TODO: Later on, adjust signature to prevent massive clutter, also rearrange as needed. May need to group some into dicts

    assert list(prompt.shape) == [1, 77, 512]  # Assuming just a single prompt, max padded for now
    assert prompt.shape == negative_prompt.shape  # Negative prompt should be empty string

    prompt = prompt.repeat(batch_size, 1, 1)
    negative_prompt = negative_prompt.repeat(batch_size, 1, 1)

    latents = torch.randn((batch_size, 4, latent_dim, latent_dim), dtype=prompt.dtype, device=device)
    combined_prompts = torch.cat((prompt, negative_prompt), dim=0)  # Should be [2, 77, 512] now
    cfg_scales = torch.tensor([cfg] * batch_size, dtype=torch.float32, device=device).reshape(batch_size, 1, 1, 1)

    # Normally steps=T, but experimentally it would be interesting to see result when skipping steps
    # It's a feature, not a bug lol
    timesteps_list = torch.linspace(start=T-1, end=0, steps=min(num_steps, T))
    timesteps_list = timesteps_list.round().long().unique_consecutive()
    for t_idx in tqdm(range(len(timesteps_list))):
        t = timesteps_list[t_idx].item()
        prev_t = timesteps_list[t_idx+1].item() if t_idx < len(timesteps_list)-1 else -1

        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)

        with torch.autocast(device_type=device, dtype=torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32):
            # Get pred from unet, apply cfg
            pred = model.unet(latents.repeat(2, 1, 1, 1), combined_prompts, timesteps.repeat(2))
            cond_output, uncond_output = pred.chunk(2, dim=0)
            pred = cfg_scales.reshape(-1, 1, 1, 1) * (cond_output - uncond_output) + uncond_output

        if model.eps_pred:
            pred_noise = pred
        else:
            alpha_bar_t = model.alpha_bar[t]
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

            pred_noise = (
                    sqrt_one_minus_alpha_bar_t * latents +
                    sqrt_alpha_bar_t * pred
            )

        # Compute mean of reverse process
        alpha_t = model.alpha[t]
        sqrt_one_minus_alpha_bar_t = model.sqrt_one_minus_alpha_bar[t]

        if use_ddpm:
            # Compute noise if t > 0 and sample the previous timestep
            noise = torch.randn_like(latents) if t > 0 else 0
            sigma_t = torch.sqrt(model.beta[t])

            latents = (latents - (model.beta[t] / sqrt_one_minus_alpha_bar_t) * pred_noise) / torch.sqrt(alpha_t) + sigma_t * noise
        else:  # DDIM
            x_hat_0 = (latents - sqrt_one_minus_alpha_bar_t * pred_noise) / model.sqrt_alpha_bar[t]
            if t > 0:
                latents = model.sqrt_alpha_bar[prev_t] * x_hat_0 + model.sqrt_one_minus_alpha_bar[prev_t] * pred_noise
            else:
                latents = x_hat_0

        # Update after this step
        if progress_callback is not None:
            progress_callback((t_idx + 1) / len(timesteps_list))

        if preview_callback and t_idx % preview_interval == 0:
            # Decode just the first image and send that back
            img = _decode_latent(vae=vae, latents=latents[:1])[0]
            preview_callback(img, t_idx)

    return _decode_latent(vae=vae, latents=latents)


def _decode_latent(vae: AutoencoderKL, latents: torch.Tensor) -> list[Image]:
    # Now decode and save the samples, shape (len_cfg_scales, 3, H, W)
    generated_images = vae.decode(latents / 0.18215).sample

    # Rescale images to [0, 1] range
    generated_images = (generated_images.clamp(-1, 1) + 1) / 2.0

    final_images = []
    for j, img_tensor in enumerate(generated_images):  # Iterate over batch dim
        # Permute to (H, W, 3) and convert to 0-255 uint8
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_uint8 = (img_np * 255).round().astype("uint8")

        # Create PIL Image and save
        img_final = Image.fromarray(img_uint8)
        final_images.append(img_final)

    return final_images


# def _update_ddpm(latents: torch.Tensor, combined_prompt: torch.Tensor):
#     pass
#
#
# def _update_ddim():
#     pass


# if __name__ == "__main__":
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     INF_CONFIG_PATH = PROJECT_ROOT / "configs/inference/default.yaml"
#
#     configs = load_inference_config()
#     print(type(configs))




