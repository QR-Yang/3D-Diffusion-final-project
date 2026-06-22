import os
from pathlib import Path
os.environ["TORCH_BLAS_PREFER_CUBLASLT"] = "0"
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from tqdm import tqdm
from data import get_dataloader
from model import PointNetUNet
batch_size = 64
num_epochs = 500
lr = 2e-5
num_train_timesteps = 8000
save_path = "checkpoints/pointnet_unet_airplain_8192*2_2e-5_8000.pth"
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader = get_dataloader(batch_size=batch_size)
    model = PointNetUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    for epoch in range(num_epochs):
        total_loss = 0
        for x0 in tqdm(dataloader):
            x0 = x0.to(device)
            noise = torch.randn_like(x0)
            t = torch.randint(0, num_train_timesteps, (x0.shape[0],), device=device)
            xt = scheduler.add_noise(x0, noise, t)
            noise_pred = model(xt, t)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print("epoch:", epoch + 1, "loss:", avg_loss)
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(model.state_dict(), save_path)
if __name__ == "__main__":
    train()