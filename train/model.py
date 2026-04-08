"""
E2-equivariant orientation model.

Input : [B, 3, H, W]  — normal map (Nx, Ny, Nz)
Output: (v_unit, v_raw)  — unit and raw 2-D irrep vector
          Predicted angle:  theta = 0.5 * atan2(v_raw[1], v_raw[0])
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from escnn import gspaces
from escnn import nn as enn


class E2DirectionIrrep(nn.Module):

    def __init__(self, N: int = 8):
        super().__init__()
        self.N = N
        self.r2_act = gspaces.rot2dOnR2(N=N)

        # (Nx, Ny) treated as a 2D vector field (irrep(1)); Nz as scalar (trivial)
        self.in_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.irrep(1), self.r2_act.trivial_repr],
        )

        c1, c2, c3, c4 = 16, 32, 64, 128
        out_type1 = enn.FieldType(self.r2_act, c1 * [self.r2_act.regular_repr])
        out_type2 = enn.FieldType(self.r2_act, c2 * [self.r2_act.regular_repr])
        out_type3 = enn.FieldType(self.r2_act, c3 * [self.r2_act.regular_repr])
        self.out_type = enn.FieldType(self.r2_act, c4 * [self.r2_act.regular_repr])

        self.block1 = enn.SequentialModule(
            enn.R2Conv(self.in_type, out_type1, kernel_size=7, padding=3, bias=False),
            enn.InnerBatchNorm(out_type1),
            enn.ReLU(out_type1),
            enn.PointwiseAvgPool(out_type1, kernel_size=2, stride=2),
        )
        self.block2 = enn.SequentialModule(
            enn.R2Conv(out_type1, out_type2, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(out_type2),
            enn.ReLU(out_type2),
            enn.PointwiseAvgPool(out_type2, kernel_size=2, stride=2),
        )
        self.block3 = enn.SequentialModule(
            enn.R2Conv(out_type2, out_type3, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(out_type3),
            enn.ReLU(out_type3),
            enn.PointwiseAvgPool(out_type3, kernel_size=2, stride=2),
        )
        self.block4 = enn.SequentialModule(
            enn.R2Conv(out_type3, self.out_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type),
        )

        self.head_out_type = enn.FieldType(self.r2_act, 1 * [self.r2_act.regular_repr])
        self.orientation_head = enn.SequentialModule(
            enn.R2Conv(self.out_type, self.head_out_type, kernel_size=1, padding=0, bias=True)
        )

        angles = 2.0 * math.pi * torch.arange(N).float() / float(N)
        self.register_buffer("angles_table", angles, persistent=False)
        self.register_buffer("wcos2", torch.cos(2.0 * angles), persistent=False)
        self.register_buffer("wsin2", torch.sin(2.0 * angles), persistent=False)

    def forward(self, x: torch.Tensor):
        x = enn.GeometricTensor(x, self.in_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        y_reg = self.orientation_head(x).tensor          # [B, N, H, W]
        B, N, H, W = y_reg.shape

        wcos = self.wcos2.view(1, N, 1, 1)
        wsin = self.wsin2.view(1, N, 1, 1)
        v2_cos = (y_reg * wcos).sum(dim=1)               # [B, H, W]
        v2_sin = (y_reg * wsin).sum(dim=1)
        irrep2_map = torch.stack([v2_cos, v2_sin], dim=1)  # [B, 2, H, W]

        v_raw = F.adaptive_avg_pool2d(irrep2_map, 1).squeeze(-1).squeeze(-1)  # [B, 2]
        v_unit = F.normalize(v_raw, dim=1, eps=1e-8)
        return v_unit, v_raw
