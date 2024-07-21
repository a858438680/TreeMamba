# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class BatchedTree:
    def __init__(self, x: torch.Tensor, indices_list: list[torch.Tensor], state_indices: list[torch.Tensor], conv_indices: torch.Tensor):
        self.x = x
        self.indices_list = indices_list
        self.state_indices = state_indices
        self.conv_indices = conv_indices

    def to(self, device: torch.device):
        return BatchedTree(
            self.x.to(device),
            [l.to(device) for l in self.indices_list],
            [l.to(device) for l in self.state_indices],
            self.conv_indices.to(device)
        )

class TreeMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            stride=4,
            kernel_size=4,
            groups=self.d_inner,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        self.learned_average = nn.Parameter(torch.ones(self.d_inner, self.d_state, device=device) * 0.5)
        self.learned_average._no_weight_decay = True

    def tree_conv(self, x, conv_indices):
        # x: (batch, d_inner, seqlen)
        # conv_indices: (batch, 4 * seqlen)

        # padding zeros to the beginning of the sequence
        x = F.pad(x, (1, 0, 0, 0, 0, 0))
        conv_indices = conv_indices.unsqueeze(1).expand(-1, self.d_inner, -1)

        # x: (batch, d_inner, seqlen + 1)
        # conv_indices: (batch, d_inner, 3 * seqlen)
        x = torch.gather(x, 2, conv_indices)

        # x: (batch, d_inner, 3 * seqlen)
        x = self.conv1d(x)

        # x: (batch, d_inner, seqlen)
        return x

    def forward(self, batched_tree: BatchedTree):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        hidden_states = batched_tree.x
        indices_list = batched_tree.indices_list
        state_indices = batched_tree.state_indices
        conv_indices = batched_tree.conv_indices
        batch, seqlen, dim = hidden_states.shape

        # conv_state, ssm_state = None, None
        # if inference_params is not None:
        #     conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
        #     if inference_params.seqlen_offset > 0:
        #         # The states are updated inplace
        #         out, _, _ = self.step(hidden_states, conv_state, ssm_state)
        #         return out

        # We do matmul and transpose BLH -> HBL at the same time
        # xz = self.in_proj(hidden_stats)
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        x, z = xz.chunk(2, dim=1)
        x = self.act(self.tree_conv(x, conv_indices))
        y = torch.zeros(batch, seqlen, dim, dtype=hidden_states.dtype, device=hidden_states.device)

        init_batch_size = indices_list[-1].shape[0]
        ssm_state = self.allocate_inference_cache(init_batch_size)
        indices = indices_list[-1]
        level_x = x[indices[:, 0], :, indices[:, 1]].unsqueeze(1)
        level_z = z[indices[:, 0], :, indices[:, 1]].unsqueeze(1)
        level_y, ssm_state = self.step(level_x, level_z, ssm_state)
        y[indices[:, 0], indices[:, 1], :] = level_y.squeeze()

        for level in reversed(range(len(indices_list) - 1)):
            indices = indices_list[level]
            states = state_indices[level]
            ssm_state = F.pad(ssm_state, (0, 0, 0, 0, 1, 0))
            left_ssm_state = ssm_state[states[:, 0], :, :]
            right_ssm_state = ssm_state[states[:, 1], :, :]
            ssm_state = self.learned_average * left_ssm_state + (1 - self.learned_average) * right_ssm_state
            level_x = x[indices[:, 0], :, indices[:, 1]].unsqueeze(1)
            level_z = z[indices[:, 0], :, indices[:, 1]].unsqueeze(1)
            level_y, ssm_state = self.step(level_x, level_z, ssm_state)
            y[indices[:, 0], indices[:, 1], :] = level_y.squeeze()

        return y

    def step(self, x, z, ssm_state):
        dtype = x.dtype
        assert x.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        x = x.squeeze(1)
        z = z.squeeze(1)
        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        # Discretize A and B
        dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        dB = torch.einsum("bd,bn->bdn", dt, B)
        ssm_state = ssm_state * dA + rearrange(x, "b d -> b d 1") * dB
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
        y = y + self.D.to(dtype) * x
        y = y * self.act(z)  # (B D)

        out = self.out_proj(y)
        return out.unsqueeze(1), ssm_state

    def allocate_inference_cache(self, batch_size, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return ssm_state
