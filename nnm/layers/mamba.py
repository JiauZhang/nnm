import torch
from torch import nn
from torch.nn import functional as F


class Mamba2Mixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.mamba_n_heads
        self.head_dim = config.mamba_d_head
        self.d_state = config.mamba_d_state
        self.d_conv = config.mamba_d_conv
        self.n_groups = config.mamba_n_groups
        self.intermediate_size = config.mamba_d_ssm
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.d_state
        self.chunk_size = config.mamba_chunk_size
        self.time_step_min, self.time_step_max = config.get("time_step_limit", (0.0, float("inf")))
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.num_heads + 1, dtype=torch.float)))
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.in_proj = nn.Linear(
            self.hidden_size, self.intermediate_size + self.conv_dim + self.num_heads, bias=config.mamba_proj_bias
        )
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.d_conv,
            groups=self.conv_dim,
            bias=config.mamba_conv_bias,
            padding=self.d_conv - 1,
        )
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.projectors_bias)
        self.act = nn.SiLU()

    def _segment_sum(self, x):
        chunk_size = x.size(-1)
        x = x[..., None].expand(*x.size(), chunk_size)
        mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=torch.bool), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=torch.bool), diagonal=0)
        return segsum.masked_fill(~mask, -torch.inf)

    def _pad_tensor(self, x, pad_size):
        if x.dim() == 3:
            return F.pad(x, (0, 0, 0, pad_size, 0, 0))
        return F.pad(x, (0, 0, 0, 0, 0, pad_size, 0, 0))

    def _reshape_chunks(self, x, pad_size, chunk_size):
        x = self._pad_tensor(x, pad_size)
        if x.dim() == 3:
            return x.reshape(x.shape[0], -1, chunk_size, x.shape[2])
        return x.reshape(x.shape[0], -1, chunk_size, x.shape[2], x.shape[3])

    def _step(self, x, B, C, dt, A, ssm_state):
        batch = x.shape[0]
        dt = dt[:, 0, :, None].expand(-1, -1, self.head_dim)
        A = A[:, None, None].expand(self.num_heads, self.head_dim, self.d_state).float()
        dA = torch.exp(dt[..., None] * A)
        B = (
            B.reshape(batch, self.n_groups, -1)[:, :, None, :]
            .expand(batch, self.n_groups, self.num_heads // self.n_groups, -1)
            .reshape(batch, -1, self.d_state)
        )
        dB = dt[..., None] * B[..., None, :]
        x_reshaped = x.reshape(batch, self.num_heads, self.head_dim)
        dBx = dB * x_reshaped[..., None]
        ssm_state = ssm_state * dA + dBx
        C = (
            C.reshape(batch, self.n_groups, -1)[:, :, None, :]
            .expand(batch, self.n_groups, self.num_heads // self.n_groups, -1)
            .reshape(batch, -1, self.d_state)
        )
        y = torch.bmm(
            ssm_state.reshape(batch * self.num_heads, self.head_dim, self.d_state),
            C.reshape(batch * self.num_heads, self.d_state, 1),
        )
        y = y.reshape(batch, self.num_heads, self.head_dim)
        D = self.D[:, None].expand(-1, self.head_dim)
        y = y + x_reshaped * D
        return y.reshape(batch, 1, -1), ssm_state

    def _scan(self, x, B, C, dt, A, seq_len):
        batch = x.shape[0]
        x = x.reshape(batch, seq_len, self.num_heads, self.head_dim).float()
        B = B.reshape(batch, seq_len, self.n_groups, self.d_state).float()
        C = C.reshape(batch, seq_len, self.n_groups, self.d_state).float()
        B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2)
        C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2)
        pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

        D_residual = self.D[None, None, :, None] * self._pad_tensor(x, pad_size)
        x = x * dt[:, :, :, None]
        A = A[None, None, :] * dt
        x, A, B, C = [self._reshape_chunks(t, pad_size, self.chunk_size) for t in (x, A, B, C)]
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)
        L = torch.exp(self._segment_sum(A))
        G = (C[:, :, :, None, :, :] * B[:, :, None, :, :, :]).sum(dim=-1)
        M = (G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]).sum(dim=-1)
        y_diag = (M[..., None] * x[:, :, None]).sum(dim=3)
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
        states = (B_decay[..., None, :] * x[..., None]).sum(dim=2)
        previous = torch.zeros_like(states[:, :1])
        states = torch.cat([previous, states], dim=1)
        decay_chunk = torch.exp(self._segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        decay_chunk = decay_chunk.transpose(1, 3)
        new_states = (decay_chunk[..., None, None] * states[:, :, None]).sum(dim=1)
        states, ssm_state = new_states[:, :-1], new_states[:, -1]
        state_decay = torch.exp(A_cumsum)
        C_states = C[..., None, :] * states[:, :, None]
        y_off = C_states.sum(-1) * state_decay.permute(0, 2, 3, 1)[..., None]
        y = y_diag + y_off
        y = y.reshape(batch, -1, self.num_heads, self.head_dim) + D_residual
        if pad_size > 0:
            y = y[:, :seq_len]
        return y.reshape(batch, seq_len, -1), ssm_state

    def forward(self, hidden, conv_state=None, ssm_state=None):
        batch, seq_len, _ = hidden.shape
        projected = self.in_proj(hidden)
        gate, hidden_bc, dt = projected.split([self.intermediate_size, self.conv_dim, self.num_heads], dim=-1)
        hidden_bc = hidden_bc.transpose(1, 2)

        if conv_state is not None and seq_len == 1:
            conv_state = torch.cat([conv_state[:, :, 1:], hidden_bc], dim=-1)
            hidden_bc = torch.sum(conv_state * self.conv1d.weight.squeeze(1), dim=-1)
            if self.conv1d.bias is not None:
                hidden_bc = hidden_bc + self.conv1d.bias
            hidden_bc = self.act(hidden_bc)
        else:
            hidden_bc = self.act(self.conv1d(hidden_bc)[..., :seq_len].transpose(1, 2))
            if conv_state is not None:
                conv_state = F.pad(hidden_bc.transpose(1, 2), (self.d_conv - hidden_bc.shape[1], 0))

        x, B, C = hidden_bc.split([self.intermediate_size, self.n_groups * self.d_state, self.n_groups * self.d_state], dim=-1)

        A = -torch.exp(self.A_log.float())
        dt = F.softplus(dt + self.dt_bias)
        dt = torch.clamp(dt, self.time_step_min, self.time_step_max)

        if conv_state is not None and seq_len == 1 and ssm_state is not None:
            y, ssm_state = self._step(x, B, C, dt, A, ssm_state)
        else:
            y, ssm_state = self._scan(x, B, C, dt, A, seq_len)
        return self.out_proj(y * F.silu(gate)), conv_state, ssm_state
