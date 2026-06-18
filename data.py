import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
class AirplaneDataset(Dataset):
    def __init__(self, path="data/pointcloud.npz", num_points=2048, length=1000):
        data = np.load(path)
        points = data["points"].astype("float32")
        center = points.mean(axis=0)
        points = points - center
        scale = np.sqrt((points ** 2).sum(axis=1)).max()
        points = points / scale
        self.points = points
        self.num_points = num_points
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        ids = np.random.choice(len(self.points), self.num_points, replace=False)
        points = self.points[ids]
        return torch.from_numpy(points)
def get_dataloader(batch_size=16):
    dataset = AirplaneDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
if __name__ == "__main__":
    dataloader = get_dataloader(batch_size=4)
    batch = next(iter(dataloader))
    print(batch.shape)
    print(batch.dtype)