import torch
import torch.nn as nn

from src.unimotion.utils import weight_init


class MLPLayer(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(MLPLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MultimodalMLPDecoder(nn.Module):
    """A naive MLP-based multimodal decoder"""

    def __init__(self, input_dim, hidden_dim, output_dim, use_intention=False) -> None:
        super().__init__()
        self.use_intention = use_intention
        if use_intention:
            self.int_proj = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, input_dim),
            )
            self.fusion_proj = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, input_dim),
            )
        else:
            self.multimodal_proj = nn.Linear(input_dim, 6 * input_dim)

        self.loc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.pi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, int_pts=None):
        n_a, n_step = x.shape[:2]
        if not self.use_intention:
            x = self.multimodal_proj(x).view(n_a, n_step, 6, -1)
        else:
            n_pts = int_pts.size(1)
            x = x.unsqueeze(-2).repeat(1, 1, n_pts, 1)
            int_feat = self.int_proj(int_pts).unsqueeze(1).repeat(1, n_step, 1, 1)
            x = self.fusion_proj(torch.cat([x, int_feat], -1))
        loc = self.loc(x)
        pi = self.pi(x).squeeze(-1)
        return loc, pi