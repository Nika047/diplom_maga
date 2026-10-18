"""
Корректор на Mamba - модели пространства состояний с селективным сканированием
(Gu, Dao, 2023), реализация на чистом PyTorch без пакета mamba-ssm (он рассчитан
на CUDA под Linux; на окне 20 обзоров последовательный проход по времени дешёвый).

Блок Mamba:
  вход -> линейный слой в две ветви (x, z);
  x -> причинная свёртка по времени (ядро MAMBA_CONV) -> SiLU;
  селективное сканирование: параметры dt, B, C зависят от входа,
      h_t = exp(dt_t A) * h_{t-1} + dt_t B_t x_t,   y_t = C_t h_t + D x_t;
  y * SiLU(z) -> линейный слой; остаточная связь и нормализация.
Блоки складываются стопкой, поправка - по выходу последнего обзора.

Проверяется на этапе 3: причинность блока (выход в момент t не зависит от входов
после t), смоук forward / backward, нулевой выход до обучения, overfit one batch.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

import config


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = config.MAMBA_D_STATE, expand: int = config.MAMBA_EXPAND,
                 d_conv: int = config.MAMBA_CONV):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner, self.d_state = d_inner, d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        # Инициализация как в статье: A = -[1..N], смещение dt - в [1e-3, 1e-1] после softplus
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.a_log = nn.Parameter(torch.log(a))
        self.d = nn.Parameter(torch.ones(d_inner))
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        b, length, _ = u.shape
        x, z = self.in_proj(u).chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))[..., :length].transpose(1, 2)  # обрезка справа = причинность
        x = F.silu(x)
        dt, bmat, cmat = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                        # (B, L, D)
        a = -torch.exp(self.a_log)                               # (D, N)
        h = torch.zeros(b, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(length):
            da = torch.exp(dt[:, t, :, None] * a)                # (B, D, N)
            h = da * h + dt[:, t, :, None] * bmat[:, t, None, :] * x[:, t, :, None]
            ys.append((h * cmat[:, t, None, :]).sum(-1) + self.d * x[:, t])
        y = torch.stack(ys, dim=1) * F.silu(z)
        return self.out_proj(y)


class MambaCorrector(nn.Module):
    def __init__(self, n_features: int, horizon: int, d_model: int = config.CORRECTOR_HIDDEN,
                 layers: int = config.CORRECTOR_LAYERS, dropout: float = config.CORRECTOR_DROPOUT):
        super().__init__()
        self.horizon = horizon
        self.inp = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon * 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Выход по всем обзорам окна (B, L, d_model) - для проверки причинности."""
        h = self.inp(x)
        for block, norm in zip(self.blocks, self.norms):
            h = h + self.drop(block(norm(h)))
        return self.norm(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.sequence(x)[:, -1]).view(-1, self.horizon, 2)
