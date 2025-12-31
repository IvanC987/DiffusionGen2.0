import math
import torch
from torch import nn
from diffusiongen2.configs import DiffusionConfig, UNetConfig
from diffusiongen2.models.unet import UNet


def create_noise_schedule(schedule: str, T: int, beta1: float, beta2: float, s: float) -> tuple:
    if schedule in ["linear", "curved"]:
        # https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L23
        # The SD implementation actually uses this curved beta as opposed to a linear one

        beta = (torch.linspace(start=beta1, end=beta2, steps=T, dtype=torch.float32) if schedule == "linear"
                else torch.linspace(start=beta1 ** 0.5, end=beta2 ** 0.5, steps=T, dtype=torch.float32) ** 2)

        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

        return beta, alpha, alpha_bar, sqrt_alpha_bar, sqrt_one_minus_alpha_bar
    elif schedule == "cosine":
        normalized_t_steps = torch.linspace(start=0, end=T, steps=T+1, dtype=torch.float32) / T
        alpha_bar_inner_term = (normalized_t_steps + s) / (1 + s) * math.pi / 2
        alpha_bar_init = torch.cos(alpha_bar_inner_term) ** 2
        alpha_bar_init = alpha_bar_init / alpha_bar_init[0]  # Normalize

        # Compute alpha, beta, and others
        alpha = alpha_bar_init[1:] / alpha_bar_init[:-1]
        alpha = torch.clamp(alpha, min=0.001, max=0.999)
        beta = 1.0 - alpha

        alpha_bar = torch.cumprod(alpha, dim=0)  # Recompute alpha bar so that it's 'linked' to clamped values
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        return beta, alpha, alpha_bar, sqrt_alpha_bar, sqrt_one_minus_alpha_bar
    else:
        raise ValueError(f"Invalid schedule provided. {schedule=}. Expected one among [linear, curved, cosine]")


class DiffusionModel(nn.Module):
    def __init__(self, diffusion_config: DiffusionConfig, unet_config: UNetConfig):
        super().__init__()

        assert diffusion_config.prediction_type in ["velocity", "epsilon"]
        assert diffusion_config.T == unet_config.T

        self.eps_pred = diffusion_config.prediction_type == "epsilon"
        self.T = diffusion_config.T


        schedule = diffusion_config.schedule
        T = diffusion_config.T
        beta1 = diffusion_config.beta1
        beta2 = diffusion_config.beta2
        s = diffusion_config.s

        scheduler_result = create_noise_schedule(schedule=schedule, T=T, beta1=beta1, beta2=beta2, s=s)
        beta, alpha, alpha_bar, sqrt_alpha_bar, sqrt_one_minus_alpha_bar = scheduler_result

        self.unet = UNet(config=unet_config)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", sqrt_alpha_bar)
        self.register_buffer("sqrt_one_minus_alpha_bar", sqrt_one_minus_alpha_bar)

    def _forward_process(self, clean_latents: torch.Tensor, timesteps: torch.Tensor):
        true_noise = torch.randn_like(clean_latents)

        # Gather the alpha values and pad for broadcasting
        sqrt_alpha_bar = self.sqrt_alpha_bar[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[timesteps].view(-1, 1, 1, 1)
        noised_latents = sqrt_alpha_bar * clean_latents + sqrt_one_minus_alpha_bar * true_noise

        return noised_latents, true_noise

    def forward(self, clean_latents: torch.Tensor, prompt: torch.Tensor, timesteps: torch.Tensor):
        assert len(clean_latents.shape) == 4, f"Expects (B, C, H, W) instead got {clean_latents.shape=}"
        assert len(prompt.shape) == 3, f"Expects (B, 77, 512) instead got {prompt.shape=}"
        assert len(timesteps.shape) == 1
        assert len(timesteps) == clean_latents.shape[0] == prompt.shape[0]  # Should all be of same B

        noised_latents, true_noise = self._forward_process(clean_latents, timesteps)
        pred = self.unet(noised_latents, prompt, timesteps)

        if self.eps_pred:
            target = true_noise
        else:
            sqrt_alpha_bar = self.sqrt_alpha_bar[timesteps].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[timesteps].view(-1, 1, 1, 1)
            target = sqrt_alpha_bar * true_noise - sqrt_one_minus_alpha_bar * clean_latents

        return pred, target

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, negative_prompts: torch.Tensor, latent_dim: int, cfg_scale: torch.Tensor):
        # `num_images` in v1.0's method shouldn't be here. In should be done in the val loop and passed in as batched prompt tensor instead
        # `num_steps` is tossed. This method is only used during validation for a few images. Would be set to `T` for now
        self.eval()

        # Keeping prompt and negative separate explicitly for safety (otherwise would assume passed in prompt contains negative)
        assert prompt.shape == negative_prompts.shape
        assert len(cfg_scale.shape) == 1  # Should just be a 1d tensor of shape (B)

        batch = prompt.shape[0]
        device = prompt.device

        # Start with complete gaussian noise (Assumes C=4 for latent channels using SD's VAE)
        latents = torch.randn((batch, 4, latent_dim, latent_dim), dtype=prompt.dtype, device=device)  # Subtle bug found by GPT: Can't start with batch*2, need to repeat in forward pass of unet
        combined_prompts = torch.cat((prompt, negative_prompts), dim=0)
        for t in range(self.T - 1, -1, -1):
            # Create timestep tensor
            timesteps = torch.full((batch,), t, device=device, dtype=torch.long)

            # Get pred from unet, apply cfg
            pred = self.unet(latents.repeat(2, 1, 1, 1), combined_prompts, timesteps.repeat(2))
            cond_output, uncond_output = pred.chunk(2, dim=0)
            pred = cfg_scale.reshape(-1, 1, 1, 1) * (cond_output - uncond_output) + uncond_output

            if self.eps_pred:
                pred_noise = pred
            else:
                alpha_bar_t = self.alpha_bar[t]
                sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

                pred_noise = (
                        sqrt_one_minus_alpha_bar_t * latents +
                        sqrt_alpha_bar_t * pred
                )

            # Compute mean of reverse process
            alpha_t = self.alpha[t]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t]

            # Compute noise if t > 0 and sample the previous timestep
            noise = torch.randn_like(latents) if t > 0 else 0
            sigma_t = torch.sqrt(self.beta[t])

            latents = (latents - (self.beta[t] / sqrt_one_minus_alpha_bar_t) * pred_noise) / torch.sqrt(alpha_t) + sigma_t * noise

        self.train()
        return latents



