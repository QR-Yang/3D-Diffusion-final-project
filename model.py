import math
import torch
import torch.nn as nn
class TimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
    def forward(self, t):
        half_dim = self.dim // 2
        device = t.device
        freqs = torch.arange(half_dim, device=device).float()
        freqs = torch.exp(-math.log(10000) * freqs / (half_dim - 1))
        t = t.float().unsqueeze(1)
        emb = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        emb = self.mlp(emb)
        return emb
class SimpleNoiseNet(nn.Module):
    def __init__(self, time_dim=128, hidden_dim=256):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(3 + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x, t):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        t_emb = self.time_embedding(t)
        t_emb = t_emb.unsqueeze(1)
        t_emb = t_emb.repeat(1, num_points, 1)
        x = torch.cat([x, t_emb], dim=2)
        noise = self.net(x)
        return noise
if __name__ == "__main__":
    model = SimpleNoiseNet()
    x = torch.randn(4, 2048, 3)
    t = torch.randint(0, 1000, (4,))
    y = model(x, t)
    print(y.shape)