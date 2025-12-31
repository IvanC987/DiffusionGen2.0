"""
Diffusion noise schedule visualization (sqrt-linear beta schedule).

Definitions:
- beta_t:
    Per-timestep noise variance. Controls how much noise is added at each step.
- alpha_t = 1 - beta_t:
    Per-timestep signal retention factor.
- alpha_bar_t = ∏_{i=1}^t alpha_i:
    Cumulative signal retention up to timestep t.
- sqrt(alpha_bar_t):
    Coefficient on the original clean image x_0 in the forward diffusion process.
- sqrt(1 - alpha_bar_t):
    Coefficient on the noise epsilon in the forward diffusion process.

Forward process:
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
"""

import torch
import matplotlib.pyplot as plt

# Number of diffusion steps
T = 1000

# Beta schedule endpoints
beta_start = 0.004
beta_end = 0.02

# Sqrt-linear beta schedule (used in LDM / Stable Diffusion)
beta = torch.linspace(
    start=beta_start ** 0.5,
    end=beta_end ** 0.5,
    steps=T,
    dtype=torch.float32
) ** 2

alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

sqrt_alpha_bar = torch.sqrt(alpha_bar)
sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(beta, label=r"$\beta_t$ (noise variance)", color="red")
plt.plot(alpha, label=r"$\alpha_t = 1 - \beta_t$", color="blue")
plt.plot(alpha_bar, label=r"$\bar{\alpha}_t$ (cumulative signal)", color="black")
plt.plot(sqrt_alpha_bar, label=r"$\sqrt{\bar{\alpha}_t}$ (signal coeff)", color="purple")
plt.plot(
    sqrt_one_minus_alpha_bar,
    label=r"$\sqrt{1 - \bar{\alpha}_t}$ (noise coeff)",
    color="green"
)

plt.xlabel("Timestep")
plt.ylabel("Value")
plt.title("Diffusion Noise Schedule (sqrt-linear β)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
