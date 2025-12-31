import math
import torch
from torch import nn
from diffusiongen2.models.attention_blocks import UNetAttentionBlock
from diffusiongen2.configs import UNetConfig


class TimeEmbedding(nn.Module):
    def __init__(self, T: int, t_embd: int):
        super().__init__()

        te = torch.zeros((T, t_embd))
        t = torch.arange(T).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, t_embd, 2).float() * -(math.log(10000.0) / t_embd))

        te[:, 0::2] = torch.sin(t / div_term)
        te[:, 1::2] = torch.cos(t / div_term)

        self.register_buffer("te", te)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.te[t]


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_groups: int,
                 t_hidden_dim: int,
                 n_embd: int,
                 n_heads: int,
                 n_layers: int,
                 dropout: float,
                 enable_flash_attention: bool):

        super().__init__()

        self.n_layers = n_layers

        self.resnet_layers = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=n_groups, num_channels=in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.Dropout(p=dropout),
            )
            for i in range(n_layers * 2)
        ])

        self.attention_blocks = nn.ModuleList([
            UNetAttentionBlock(channels=out_channels,
                               n_groups=n_groups,
                               n_embd=n_embd,
                               n_heads=n_heads,
                               enable_flash_attention=enable_flash_attention
                               )
            for _ in range(n_layers)])

        self.t_embd_linear = nn.Linear(t_hidden_dim, out_channels, bias=False)
        self.residual_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.downsample = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor, t: torch.Tensor):
        """
        :param x: Input tensor (B, C, H, W)
        :param prompt: Prompt tensor from (FG) CLIP (B, 77, 512)
        :param t: Raw timestep tensor (B, t_embd)  # DiffusionGen1.0 was wrong about shape?
        :return: Downsampled latent image tensor along with skip connection
        """

        # TODO: Verify input tensor dimensions and shape of h over time
        # (B, t_embd) -> (B, out_channels, 1, 1)
        # Convert to match out_channels and unsqueeze for element-wise addition + broadcasting
        t_embd = self.t_embd_linear(t).unsqueeze(-1).unsqueeze(-1)

        h = x
        for i in range(self.n_layers):
            res = h  # Initial residual connection
            h = self.resnet_layers[i](h)
            h = h + t_embd

            # Add residual and stabilize gradient
            # Bug Fix from V1.0: Only scaled residual path in previous version
            res_path = self.residual_conv(res) if i == 0 else res
            h = (h + res_path) / (2 ** 0.5)

            res = h  # Midpoint residual

            h = self.resnet_layers[self.n_layers + i](h)
            h = self.attention_blocks[i](h, prompt)

            h = (h + res) / (2 ** 0.5)

        return self.downsample(h), h


class BottleNeck(nn.Module):
    def __init__(self,
                 channels: int,
                 n_groups: int,
                 t_hidden_dim: int,
                 n_embd: int,
                 n_heads: int,
                 n_layers: int,
                 dropout: float,
                 enable_flash_attention: bool):

        super().__init__()

        self.n_layers = n_layers

        self.resnet_layers = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=n_groups, num_channels=channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
                nn.Dropout(p=dropout),
            )
            for _ in range(n_layers * 2)
        ])

        self.t_embd_linear = nn.Linear(t_hidden_dim, channels, bias=False)

        self.attention_blocks = nn.ModuleList([
            UNetAttentionBlock(channels=channels,
                               n_groups=n_groups,
                               n_embd=n_embd,
                               n_heads=n_heads,
                               enable_flash_attention=enable_flash_attention
                               )
            for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, prompt: torch.Tensor, t: torch.Tensor):
        # TODO: Test out input tensor shapes

        t_embd = self.t_embd_linear(t).unsqueeze(-1).unsqueeze(-1)

        h = x
        for i in range(self.n_layers):
            res = h
            h = self.resnet_layers[i](h)
            h = h + t_embd
            h = (h + res) / (2 ** 0.5)

            res = h
            h = self.resnet_layers[self.n_layers + i](h)
            h = self.attention_blocks[i](h, prompt)
            h = (h + res) / (2 ** 0.5)

        return h


class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_groups: int,
                 t_hidden_dim: int,
                 n_embd: int,
                 n_heads: int,
                 n_layers: int,
                 dropout: float,
                 enable_flash_attention: bool):

        super().__init__()

        self.n_layers = n_layers

        self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        self.skip_conv = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.t_embd_linear = nn.Linear(t_hidden_dim, out_channels, bias=False)

        self.resnet_layers = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=n_groups, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.Dropout(p=dropout)
            )
            for i in range(n_layers * 2)
        ])

        self.attention_blocks = nn.ModuleList([
            UNetAttentionBlock(channels=out_channels,
                               n_groups=n_groups,
                               n_embd=n_embd,
                               n_heads=n_heads,
                               enable_flash_attention=enable_flash_attention
                               )
            for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor, prompt: torch.Tensor, t: torch.Tensor):
        # TODO: Verify the shapes of all these incoming tensors!

        h = x
        h = self.upsample(h)
        h = torch.cat((h, skip), dim=1)  # Concat along channels dim
        h = self.skip_conv(h)

        t_embd = self.t_embd_linear(t).unsqueeze(-1).unsqueeze(-1)
        for i in range(self.n_layers):
            res = h
            h = self.resnet_layers[i](h)
            h = h + t_embd

            h = (h + res) / (2 ** 0.5)

            res = h
            h = self.resnet_layers[self.n_layers + i](h)
            h = self.attention_blocks[i](h, prompt)
            h = (h + res) / (2 ** 0.5)

        return h


class UNet(nn.Module):
    def __init__(self, config: UNetConfig):
        """
        UNet model - core network used in the diffusion process
        All hyperparameters are provided via UNetConfig
        """
        super().__init__()

        in_channels = config.in_channels        # Input channels (e.g. 3 for RGB, 4 for latent space)
        channels = config.channels              # Channel progression per resolution level
        n_layers = config.n_layers              # Number of residual layers per block
        n_bn = config.n_bn                      # Number of bottleneck layers
        n_groups = config.n_groups              # Number of groups for GroupNorm
        T = config.T                            # Maximum diffusion timesteps (e.g. 1000)
        t_embd = config.t_embd                  # Base timestep embedding dimension
        t_embd_scale = config.t_embd_scale      # MLP expansion factor for timestep embedding
        n_embd = config.n_embd                  # Prompt embedding dim (e.g. 512 for CLIP)
        n_heads = config.n_heads                # Number of attention heads
        enable_flash_attention = config.enable_flash_attention  # Whether to use FlashAttention
        dropout = config.dropout                # Dropout probability

        assert t_embd_scale > 0

        self.time_embd_layer = TimeEmbedding(T=T, t_embd=t_embd)
        t_hidden_dim = t_embd * t_embd_scale  # Hidden dim used inside timestep MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(t_embd, t_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(t_hidden_dim, t_hidden_dim, bias=True)
        )

        self.in_conv = nn.Conv2d(in_channels=in_channels, out_channels=channels[0], kernel_size=3, stride=1, padding=1)

        self.encoders = nn.ModuleList([
            Encoder(in_channels=channels[i],
                    out_channels=channels[i+1],
                    n_groups=n_groups,
                    t_hidden_dim=t_hidden_dim,
                    n_embd=n_embd,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    dropout=dropout,
                    enable_flash_attention=enable_flash_attention)
            for i in range(len(channels)-1)
        ])

        self.bottle_necks = nn.ModuleList([
            BottleNeck(channels=channels[-1],
                       n_groups=n_groups,
                       t_hidden_dim=t_hidden_dim,
                       n_embd=n_embd,
                       n_heads=n_heads,
                       n_layers=n_layers,
                       dropout=dropout,
                       enable_flash_attention=enable_flash_attention)
            for _ in range(n_bn)
        ])

        self.decoders = nn.ModuleList([
            Decoder(in_channels=channels[i],
                    out_channels=channels[i-1],
                    n_groups=n_groups,
                    t_hidden_dim=t_hidden_dim,
                    n_embd=n_embd,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    dropout=dropout,
                    enable_flash_attention=enable_flash_attention)
            for i in range(len(channels)-1, 0, -1)
        ])

        self.final_layer = nn.Sequential(
            nn.GroupNorm(num_groups=n_groups, num_channels=channels[0]),
            nn.SiLU(),
            nn.Conv2d(in_channels=channels[0], out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor, prompt: torch.Tensor, t: torch.Tensor):
        """
        Given input image, prompt, and timestep tensors, returns output tensor of same shape representing predicted noise

        :param x: Input image/latent tensor of shape (b, c, h, w)
        :param prompt: Prompt tensor of shape (b, 77, 512)
        :param t: Timestep tensor of shape (b)
        :return: Returns predicted noise in image
        """
        # TODO: Confirm above copied description from v1.0 is correct later

        assert len(x.shape) == 4, f"Given tensor x should be of shape (b, c, h, w), instead got {x.shape}"
        assert list(prompt.shape) == [x.shape[0], 77, 512], f"Prompt tensor should be of shape ({x.shape[0]}, 77, 512) but got {prompt.shape=}"
        assert len(t) == x.shape[0], f"Expected num timesteps == batch_size, {len(t)=}, {x.shape[0]=}"
        assert len(t.shape) == 1, "timesteps t should be a vector of shape (b)"

        ts = self.time_embd_layer(t)
        ts_hidden = self.time_mlp(ts)

        h = self.in_conv(x)

        skip_connections = []
        for enc in self.encoders:
            h, skip = enc(h, prompt, ts_hidden)
            skip_connections.append(skip)

        for bn in self.bottle_necks:
            h = bn(h, prompt, ts_hidden)

        for dec in self.decoders:
            h = dec(h, skip_connections.pop(), prompt, ts_hidden)

        return self.final_layer(h)

