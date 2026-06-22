from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
num_points = 8192*2
class PointDataset(Dataset):
    def __init__(self, root="data/airplane_points"):
        self.files = sorted(Path(root).glob("*.npy"))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        points = np.load(self.files[index]).astype("float32")
        choice = np.random.choice(points.shape[0], num_points, replace=False)
        points = points[choice]
        return torch.from_numpy(points)
def get_dataloader(batch_size=1):
    dataset = PointDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
if __name__ == "__main__":
    dataset = PointDataset()
    dataloader = get_dataloader(batch_size=32)
    batch = next(iter(dataloader))