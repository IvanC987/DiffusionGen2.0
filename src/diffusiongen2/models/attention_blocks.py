import torch
from torch import nn


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, n_heads: int, enable_flash_attention: bool):
        super().__init__()
        assert channels % n_heads == 0

        self.enable_flash_attention = enable_flash_attention
        self.n_heads = n_heads

        self.qkv = nn.Linear(channels, 3 * channels, bias=False)
        self.wo = nn.Linear(channels, channels, bias=False)

    def forward(self, x: torch.Tensor):
        # x.shape == (B, HW, C)
        B, HW, C = x.shape

        # (B, HW, C) -> (B, HW, n_heads, h_dim=C//n_heads)
        qkv = self.qkv(x).view(B, HW, self.n_heads, C//self.n_heads * 3)

        # Each is (B, n_heads, HW, h_dim)
        q, k, v = qkv.permute(0, 2, 1, 3).chunk(3, dim=-1)

        if self.enable_flash_attention:
            output = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            raw_attention = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)
            attention_scores = torch.softmax(raw_attention, dim=-1)
            output = attention_scores @ v

        # (B, n_heads, HW, h_dim)
        output = output.permute(0, 2, 1, 3).reshape(B, HW, C)

        return self.wo(output)


class CrossAttentionBlock(nn.Module):
    def __init__(self, channels: int, n_embd: int, n_heads: int, enable_flash_attention: bool):
        super().__init__()
        assert channels % n_heads == 0

        self.n_embd = n_embd
        self.n_heads = n_heads
        self.enable_flash_attention = enable_flash_attention

        self.q =  nn.Linear(channels, channels, bias=False)
        self.kv = nn.Linear(n_embd, channels * 2, bias=False)
        self.wo = nn.Linear(channels, channels, bias=False)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor):
        # x.shape ==      (B, HW, C)
        # prompt.shape == (B, 77, n_embd=512)
        assert list(prompt.shape)[1:] == [77, self.n_embd]

        B, HW, C = x.shape

        # (B, HW, C) -> (B, n_heads, HW, h_dim)
        q = self.q(x).reshape(B, HW, self.n_heads, C//self.n_heads).permute(0, 2, 1, 3)

        # (B, 77, n_embd) -> (B, 77, C) -> (B, 77, n_heads, h_dim) -> (B, n_heads, 77, h_dim)
        k, v = self.kv(prompt).reshape(B, 77, self.n_heads, C//self.n_heads * 2).permute(0, 2, 1, 3).chunk(2, dim=-1)

        if self.enable_flash_attention:
            output = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # (B, n_heads, HW, h_dim) @ (B, n_heads, h_dim, 77) -> (B, n_heads, HW, 77)
            raw_attention = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)
            attention_scores = torch.softmax(raw_attention, dim=-1)
            # (B, n_heads, HW, 77) @ (B, n_heads, 77, h_dim) -> (B, n_heads, HW, h_dim)
            output = attention_scores @ v

        # (B, n_heads, HW, h_dim) -> (B, HW, C)
        output = output.permute(0, 2, 1, 3).reshape(B, HW, C)

        return self.wo(output)


class UNetAttentionBlock(nn.Module):
    """
    This serves as a single attention block by combining self attention, cross attention, gn/ln layers and geglu
    Felt like the one shown by Umar Jamil works very well, so I just borrowed this block with minor changes
    """

    def __init__(self, channels: int, n_groups: int, n_embd: int, n_heads: int, enable_flash_attention: bool):
        """
        Combines the entire attention portion into a single block for modularity.
        Uses MHSA, MHCA, Conv layers, Residual connection, GeGLU and such.

        :param channels: Number of channels of input tensor
        :param n_groups: Number of normalization groups for groupnorm
        :param n_embd: Number of embedding dimension (Fixed at 512 for CLIP)
        :param n_heads: Number of attention heads
        :param enable_flash_attention: Boolean to allow usage of flash attention
        """

        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups=n_groups, num_channels=channels)
        self.conv_input = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0)

        self.ln1 = nn.LayerNorm(channels)
        self.self_attention = SelfAttentionBlock(channels=channels, n_heads=n_heads, enable_flash_attention=enable_flash_attention)
        self.ln2 = nn.LayerNorm(channels)
        self.cross_attention = CrossAttentionBlock(channels=channels, n_embd=n_embd, n_heads=n_heads, enable_flash_attention=enable_flash_attention)
        self.ln3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2, bias=False)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels, bias=False)

        self.conv_output = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor):
        # x:      (B, C, H, W)
        # prompt: (B, 77, 512)

        # TODO: Test out adding /(2 ** 0.5) norm to the short residual addition later on and test diff!

        residual_long = x

        h = self.groupnorm(x)
        h = self.conv_input(h)

        # Take the input image tensor and transform into text-like shape of (B, H*W, C), analogous to (B, seq_len, n_embd)
        B, C, H, W = h.shape
        h = h.view(B, C, H*W)
        h = h.permute(0, 2, 1)

        residual_short = h
        h = self.ln1(h)
        h = self.self_attention(h)

        # ----------
        h = h + residual_short
        residual_short = h

        h = self.ln2(h)
        h = self.cross_attention(h, prompt)

        h = h + residual_short
        residual_short = h

        h = self.ln3(h)
        # ----------

        # "GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10"
        # After linear_geglu_1, h.shape -> (B, HW, 8 * C) then split into two tensors of shape (B, HW, 4 * C)
        h, gate = self.linear_geglu_1(h).chunk(2, dim=-1)
        h = h * nn.functional.gelu(gate)

        # (B, HW, 4 * C) -> (B, HW, C)
        h = self.linear_geglu_2(h)
        h = h + residual_short

        # ((B, HW, C) -> (B, C, H, W)
        h = h.permute(0, 2, 1)
        h = h.view(B, C, H, W)

        # Finally, conv + residual connection and return
        return self.conv_output(h) + residual_long

