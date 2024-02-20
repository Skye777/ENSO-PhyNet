#reference: https://github.com/NVlabs/AFNO-transformer
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.realpath(".."))
sys.path.append('/home/dl/Desktop/vit/')

from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from FNO.utilities3 import *
#from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
import torch.fft
from einops import rearrange, repeat
from pos_embed import  get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO1D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape

        x = torch.fft.rfft(x, dim=1, norm="ortho")
        x = x.reshape(B, N // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, N // 2 + 1, C)
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")
        x = x.type(dtype)
        kernel_vis = [self.w1, self.b1, self.w2, self.b2]

        return x + bias, kernel_vis
    

class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)


        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1,2), norm="ortho")
        x = x.type(dtype)

        kernel_vis = [self.w1, self.b1, self.w2, self.b2]

        return x + bias, kernel_vis


class Block(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_heads = 4,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.time_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.time_attn = AFNO1D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction)
        self.space_attn = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        B, H, W, T, C = x.shape

        residual = x
        x = self.norm1(x)
        x = x.reshape(B*H*W, T, C)
        x, time_attn_map = self.time_attn(x)
        x, space_kernel_attn = self.space_attn(x.reshape(B, H, W, T, C).permute(0, 3, 1, 2, 4).reshape(B*T, H, W, C))
        x = x.reshape(B, T, H, W, C).permute(0, 2, 3, 1, 4)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)

        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        
        return x, time_attn_map, space_kernel_attn


class LinearCrossAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, n_hb_term):
        super(LinearCrossAttention, self).__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.query = nn.Linear(n_embd, n_embd)
        self.keys = nn.ModuleList([nn.Linear(n_embd, n_embd) for _ in range(n_hb_term)])
        self.values = nn.ModuleList([nn.Linear(n_embd, n_embd) for _ in range(n_hb_term)])
        # regularization
        # self.attn_drop = nn.Dropout(config.attn_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.n_hb_term = n_hb_term

        self.attn_type = 'l1'

    '''
        Linear Attention and Linear Cross Attention (if y is provided)
    '''
    def forward(self, x, y=None, layer_past=None):
        y = x if y is None else y
        B, N1, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, N1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.softmax(dim=-1)
        out = q
        for i in range(self.n_hb_term):
            _, T2, _ = y[i].size()
            k = self.keys[i](y[i]).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            v = self.values[i](y[i]).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            k = k.softmax(dim=-1)  #
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)  # normalized
            out = out +  1 * (q @ (k.transpose(-2, -1) @ v)) * D_inv


        # output projection
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        return out


class AFNONet(nn.Module):
    def __init__(
            self,
            img_size=(720, 1440),
            patch_size=(16, 16),
            in_chans=2,
            out_chans=2,
            default_vars=7,
            embed_dim=384,
            num_heads=4,
            depth=2,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        ):
        super().__init__()
        # self.params = params
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.default_vars = default_vars
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks 
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # self.patch_embeds = nn.ModuleList([PatchEmbed(patch_size=self.patch_size, embed_dim=embed_dim) for _ in range(default_vars)])
        self.patch_embed = PatchEmbed(patch_size=self.patch_size, embed_dim=embed_dim//2)

        self.var_agg = LinearCrossAttention(n_embd=embed_dim//2, n_head=num_heads, n_hb_term=default_vars-1)
        # self.var_agg = nn.MultiheadAttention(embed_dim//2, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim//2)

        # 1. temporal embedding
        self.time_embed = nn.Parameter(torch.zeros(1, self.in_chans, embed_dim), requires_grad=True)
        # 2. spatial embedding
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.linear_embed = nn.Linear(embed_dim//2, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_heads = num_heads,
            num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction) 
        for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, (self.out_chans//self.in_chans)*self.patch_size[0]*self.patch_size[1], bias=False)
        self.sig = nn.Sigmoid()

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size[0]),
            int(self.img_size[1] / self.patch_size[1]),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        time_embed = get_1d_sincos_pos_embed_from_grid(self.time_embed.shape[-1], np.arange(self.in_chans))
        self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        # token embedding layer
        # for i in range(len(self.patch_embeds)):
        #     w = self.patch_embeds[i].proj.weight.data
        #     trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
        w = self.patch_embed.proj.weight.data
        trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x_emb = [self.patch_embed(x[..., i]) for i in range(x.shape[-1])]
        x1 = x_emb[0].reshape(B, self.h, self.w, self.in_chans, self.embed_dim//2)
        x1_edge = x1.clone()
        x1_edge[:, 5:-5, 10:, ...] = 0

        x2 = [x2_i.reshape(B, self.h, self.w, self.in_chans, self.embed_dim//2) for x2_i in x_emb[1:]]
        x2 = [x2_i[:, 5:-5, 10:, ...].flatten(1, 2).flatten(1, 2) for x2_i in x2]
        # x2 = torch.cat(x2, dim=1)
        mx = self.var_agg(x1[:, 5:-5, 10:, ...].flatten(1, 3), x2)
        x1_edge[:, 5:-5, 10:, ...] = mx.reshape(B, self.h-10, self.w-10, self.in_chans, self.embed_dim//2)
        x = self.norm1(x1+x1_edge)
        
        x = self.linear_embed(x.flatten(1, 2)) + self.pos_embed.unsqueeze(2) + self.time_embed.unsqueeze(1)
        
        x = x.reshape(B, self.h, self.w, self.in_chans, self.embed_dim)
        timeAttn = []
        spaceAttn = []
        for blk in self.blocks:
            x, t, s = blk(x)
            timeAttn.append(t)
            spaceAttn.append(s)
        # print(x2.shape)

        return x, timeAttn, spaceAttn

    def forward(self, x):
        x, timeAttn, spaceAttn = self.forward_features(x)

        # B, H, W, T, C = x.shape
        # x = x.reshape(B, H, W, T*C)

        x = self.head(x)
        x = rearrange(
            x,
            "b h w t (p1 p2 c_out) -> b (t c_out) (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        x = self.sig(x)
        return x, timeAttn, spaceAttn


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=(16, 16), embed_dim=768):
        super().__init__()
        self.proj = nn.Conv3d(1, embed_dim, kernel_size=(1, patch_size[0], patch_size[1]), stride=(1, patch_size[0], patch_size[1]))

    def forward(self, x):
        # B, T, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x.unsqueeze(1)).flatten(3).transpose(1, 3)
        return x


if __name__ == "__main__":
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    model = AFNONet(img_size=(60, 160), patch_size=(4, 4), in_chans=6, out_chans=18).to(device)
    sample = torch.randn(1, 6, 60, 160, 7).to(device)
    result, _, _, = model(sample)
    print(count_params(model))
    print(result.shape)
    print(torch.norm(result))
    # model = Highway(128, 2, f=torch.nn.functional.relu).to(device)
    # input = torch.randn(1, 15, 40, 6, 128).to(device)
    # out = model(input)
    # print(out.shape)
    # x = torch.randn(1, 64, 15, 40, 6).to(device)
    # unet = U_net(64, 64, 3, 0).to(device)
    # y = unet(x)
    # print(y.shape)
    # print(count_params(unet))
    
