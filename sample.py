import os
from pathlib import Path
os.environ["TORCH_BLAS_PREFER_CUBLASLT"] = "0"
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"
import torch
from diffusers import DDPMScheduler
from model import PointNetUNet
num_points = 4096*4
num_train_timesteps = 8000
num_inference_steps = 8000
checkpoint_path = "checkpoints/pointnet_unet_airplain_8192*2_2e-5_8000.pth"
save_dir = "outputs"
def save_ply(points, path):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
def sample():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PointNetUNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    scheduler.set_timesteps(num_inference_steps)
    x = torch.randn(1, num_points, 3).to(device)
    with torch.no_grad():
        for t in scheduler.timesteps:
            timestep = torch.tensor([t], device=device)
            noise_pred = model(x, timestep)
            x = scheduler.step(noise_pred, t, x).prev_sample
    points = x[0].cpu().numpy()
    Path(save_dir).mkdir(exist_ok=True)
    save_ply(points, f"{save_dir}/sample_chair.ply")
    print("saved:", f"{save_dir}/sample_chair.ply")
if __name__ == "__main__":
    sample()