from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from tqdm import tqdm
from data import get_dataloader
from model import SimpleNoiseNet
batch_size = 16
num_epochs = 50
lr = 1e-4
num_train_timesteps = 1000
save_path = "checkpoints/simple_noise_net.pth"
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader = get_dataloader(batch_size=batch_size)
    model = SimpleNoiseNet().to(device)
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