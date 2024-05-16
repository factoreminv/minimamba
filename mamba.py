import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
#import mlx.nn as nn


DEVICE = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Mamba(nn.Module):
    def __init__(self, d_model, n_layers, vocab_size, d_state, d_conv, expand, dt_rank):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank

        self.embedding = nn.Embedding(vocab_size, d_model).to(DEVICE)
        self.layers = list([ResidualBlock(d_model, n_layers, vocab_size, d_state, d_conv, expand, dt_rank).to(DEVICE) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)

        self.out_head = nn.Linear(d_model, vocab_size, bias=False).to(DEVICE)
        self.out_head.weight = self.embedding.weight

    def forward(self, input_sequence_ids):
        e = self.embedding(input_sequence_ids)
        for l in self.layers:
            e = l(e)
        e = self.norm(e)
        return self.out_head(e)

class ResidualBlock(nn.Module):
    def __init__(self, d_model, n_layers, vocab_size, d_state, d_conv, expand, dt_rank):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank

        self.mambablock = MambaBlock(d_model, n_layers, vocab_size, d_state, d_conv, expand, dt_rank)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        return self.mambablock(self.norm(x)) + x

class MambaBlock(nn.Module):
    def __init__(self, d_model, n_layers, vocab_size, d_state, d_conv, expand, dt_rank):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank

        self.res_x = nn.Linear(d_model, d_model*expand)
        self.state_x = nn.Linear(d_model, d_model*expand)

        self.conv = nn.Conv1d(
            in_channels=d_model*expand,
            out_channels=d_model*expand,
            bias=False,
            kernel_size=d_conv,
            padding=d_conv - 1,
        )

        self.x_to_dt = nn.Linear(d_model*expand, dt_rank)
        self.x_to_B = nn.Linear(d_model*expand, d_state)
        self.x_to_C = nn.Linear(d_model*expand, d_state)

        A = einops.repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_model*expand)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model*expand))
        self.out_proj = nn.Linear(d_model*expand, d_model)

        self.dt_proj = nn.Linear(dt_rank, d_model*expand, bias=True)

    def forward(self, x):
        b, l, d = x.shape

        res = self.res_x(x)
        x = self.state_x(x)

        x = einops.rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv(x)[:, :, :l]
        x = einops.rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)

        return self.out_proj(y)

    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        delta = F.softplus(self.dt_proj(self.x_to_dt(x)))
        B = self.x_to_B(x)
        C = self.x_to_C(x)

        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einops.einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einops.einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einops.einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        return y + u * D

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

if __name__ == "__main__":
    x=1
    x=x
