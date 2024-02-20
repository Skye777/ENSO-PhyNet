import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from copy import deepcopy


class make_embedding(nn.Module):
    def __init__(
        self,
        cube_dim,
        d_size,
        emb_spatial_size,
        max_len,
        device,
    ):
        """
        :param cube_dim: The number of grids in one patch cube
        :param d_size: the embedding length
        :param emb_spatial_size:The number of patches decomposed in a field, S
        :param max_len: look back or prediction length, T
        """
        super().__init__()
        # 1. temporal embedding
        pe = torch.zeros(max_len, d_size)
        temp_position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_size, 2) * -(np.log(10000.0) / d_size))
        pe[:, 0::2] = torch.sin(temp_position * div_term)
        pe[:, 1::2] = torch.cos(temp_position * div_term)
        self.pe_time = pe[None, None].to(device)
        # 2. spatial embedding
        self.spatial_pos = torch.arange(emb_spatial_size)[None, :, None].to(device)   # (1, 510, 1)
        self.emb_space = nn.Embedding(emb_spatial_size, d_size)   # (1, 510, 1, 256)
        self.linear = nn.Linear(cube_dim, d_size)
        self.norm = nn.LayerNorm(d_size)

    def forward(self, x):
        assert len(x.size()) == 4
        embedded_space = self.emb_space(self.spatial_pos)

        # (4, 510, 12, 256)   (1, 1, 12, 256)   (1, 510, 1, 256)
        x = self.linear(x) + self.pe_time[:, :, : x.size(2)] + embedded_space
        return self.norm(x)
    

if __name__ == "__main__":

    device = "cpu"
    predictor_emb = make_embedding(cube_dim=108, d_size=256, emb_spatial_size=510, max_len=12, device=device)
    sample = torch.randn(4, 510, 12, 108)
    result = predictor_emb(sample)
    print(result.shape)
    # print(torch.norm(result))
