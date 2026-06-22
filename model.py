import math
import torch
import torch.nn as nn
def mlp(channels):
    if len(channels) == 3:
        net = nn.Sequential(
            nn.Linear(channels[0], channels[1]),
            nn.SiLU(),
            nn.Linear(channels[1], channels[2]),
        )
    else:
        net = nn.Sequential(
            nn.Linear(channels[0], channels[1]),
            nn.SiLU(),
            nn.Linear(channels[1], channels[2]),
            nn.SiLU(),
            nn.Linear(channels[2], channels[3]),
        )
    return net
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        )
    def forward(self, x):
        return nn.SiLU()(x + self.net(x))
class TimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.net = mlp([dim, dim * 2, dim*2, dim])
    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.arange(half_dim, device=t.device).float()
        freqs = torch.exp(-math.log(10000) * freqs / (half_dim - 1))
        t = t.float().unsqueeze(1)
        x = t * freqs.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        x = self.net(x)
        return x
def gather_points(x, index):
    channels = x.shape[2]
    index = index.unsqueeze(2)
    index = index.repeat(1, 1, channels)
    out = torch.gather(x, 1, index)
    return out
def gather_neighbors(x, index):
    batch_size = x.shape[0]
    batch_index = torch.arange(batch_size, device=x.device)
    batch_index = batch_index.view(batch_size, 1, 1)
    batch_index = batch_index.repeat(1, index.shape[1], index.shape[2])
    out = x[batch_index, index]
    return out
def fps_sample(pos, num_sample):
    batch_size = pos.shape[0]
    num_points = pos.shape[1]
    device = pos.device
    sample_index = torch.zeros(batch_size, num_sample, dtype=torch.long, device=device)
    distance = torch.ones(batch_size, num_points, device=device) * 1e10
    farthest = torch.randint(0, num_points, (batch_size,), device=device)
    batch_index = torch.arange(batch_size, device=device)
    for i in range(num_sample):
        sample_index[:, i] = farthest
        center = pos[batch_index, farthest]
        center = center.view(batch_size, 1, 3)
        dist = ((pos - center) ** 2).sum(dim=2)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = distance.max(dim=1).indices
    return sample_index
class DownBlock(nn.Module):
    def __init__(self, num_sample, k_list, in_channels, out_channels):
        super().__init__()
        self.num_sample = num_sample
        self.k_list = k_list
        self.max_k = max(k_list)
        branch_channels = out_channels // 2
        self.branch1 = mlp([in_channels + 3, branch_channels, branch_channels])
        self.branch2 = mlp([in_channels + 3, branch_channels, branch_channels])
        self.branch3 = mlp([in_channels + 3, branch_channels, branch_channels])
        self.fuse = mlp([branch_channels * 3, out_channels, out_channels])
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)
    def forward(self, x, pos):
        sample_index = fps_sample(pos, self.num_sample)
        new_pos = gather_points(pos, sample_index)
        dist = torch.cdist(new_pos, pos)
        neighbor_index = dist.topk(self.max_k, largest=False).indices
        neighbor_x = gather_neighbors(x, neighbor_index)
        neighbor_pos = gather_neighbors(pos, neighbor_index)
        center_pos = new_pos.unsqueeze(2)
        relative_pos = neighbor_pos - center_pos
        feature = torch.cat([neighbor_x, relative_pos], dim=3)
        k1 = self.k_list[0]
        k2 = self.k_list[1]
        k3 = self.k_list[2]
        f1 = self.branch1(feature[:, :, :k1, :]).max(dim=2).values
        f2 = self.branch2(feature[:, :, :k2, :]).max(dim=2).values
        f3 = self.branch3(feature[:, :, :k3, :]).max(dim=2).values
        out = torch.cat([f1, f2, f3], dim=2)
        out = self.fuse(out)
        out = self.res1(out)
        out = self.res2(out)
        return out, new_pos
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = mlp([in_channels, out_channels, out_channels])
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)
    def forward(self, small_x, small_pos, big_x, big_pos):
        dist = torch.cdist(big_pos, small_pos)
        index = dist.topk(3, largest=False).indices
        neighbor_x = gather_neighbors(small_x, index)
        neighbor_dist = dist.gather(2, index)
        weight = 1.0 / (neighbor_dist + 1e-8)
        weight = weight / weight.sum(dim=2, keepdim=True)
        x = neighbor_x * weight.unsqueeze(3)
        x = x.sum(dim=2)
        x = torch.cat([x, big_x], dim=2)
        x = self.net(x)
        x = self.res1(x)
        x = self.res2(x)
        return x
class PointNetUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(128)
        self.input_mlp = mlp([3 + 128, 128, 128])
        self.input_res = ResidualBlock(128)
        self.down1 = DownBlock(512, [16, 32, 64], 128, 256)
        self.down2 = DownBlock(128, [16, 32, 64], 256, 512)
        self.down3 = DownBlock(32, [16, 32, 64], 512, 768)
        self.time0 = nn.Linear(128, 128)
        self.time1 = nn.Linear(128, 256)
        self.time2 = nn.Linear(128, 512)
        self.time3 = nn.Linear(128, 768)
        self.up3 = UpBlock(768 + 512, 512)
        self.up2 = UpBlock(512 + 256, 256)
        self.up1 = UpBlock(256 + 128, 128)
        self.output_mlp = mlp([128, 128, 64, 3])
    def forward(self, x, t):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        pos0 = x
        t_emb = self.time_embedding(t)
        t0 = t_emb.unsqueeze(1).repeat(1, num_points, 1)
        x0 = torch.cat([x, t0], dim=2)
        x0 = self.input_mlp(x0)
        x0 = x0 + self.time0(t_emb).unsqueeze(1)
        x0 = self.input_res(x0)
        x1, pos1 = self.down1(x0, pos0)
        x1 = x1 + self.time1(t_emb).unsqueeze(1)
        x2, pos2 = self.down2(x1, pos1)
        x2 = x2 + self.time2(t_emb).unsqueeze(1)
        x3, pos3 = self.down3(x2, pos2)
        x3 = x3 + self.time3(t_emb).unsqueeze(1)
        x2 = self.up3(x3, pos3, x2, pos2)
        x1 = self.up2(x2, pos2, x1, pos1)
        x0 = self.up1(x1, pos1, x0, pos0)
        out = self.output_mlp(x0)
        return out
if __name__ == "__main__":
    model = PointNetUNet()
    x = torch.randn(1, 4096, 3)
    t = torch.randint(0, 5000, (1,))
    y = model(x, t)
    print(y.shape)