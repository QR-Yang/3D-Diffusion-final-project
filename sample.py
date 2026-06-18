from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler
from trimesh.voxel.ops import matrix_to_marching_cubes

from model import SimpleNoiseNet


num_points = 2048
num_train_timesteps = 1000
checkpoint_path = "checkpoints/simple_noise_net.pth"
save_dir = "outputs"
grid_size = 64


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


def points_to_mesh(points, path):
    points = points - points.mean(axis=0)
    points = points / np.abs(points).max()

    points = (points + 1) / 2
    ids = (points * (grid_size - 1)).astype("int")

    grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    grid[ids[:, 0], ids[:, 1], ids[:, 2]] = True

    mesh = matrix_to_marching_cubes(grid, pitch=2.0 / grid_size)
    mesh.vertices = mesh.vertices - 1
    mesh.export(path)


def sample():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleNoiseNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    scheduler.set_timesteps(num_train_timesteps)

    x = torch.randn(1, num_points, 3).to(device)

    with torch.no_grad():
        for t in scheduler.timesteps:
            timestep = torch.tensor([t], device=device)
            noise_pred = model(x, timestep)
            x = scheduler.step(noise_pred, t, x).prev_sample

    points = x[0].cpu().numpy()

    Path(save_dir).mkdir(exist_ok=True)

    np.save(f"{save_dir}/sample.npy", points)
    save_ply(points, f"{save_dir}/sample.ply")
    points_to_mesh(points, f"{save_dir}/sample.obj")

    print("saved:", f"{save_dir}/sample.npy")
    print("saved:", f"{save_dir}/sample.ply")
    print("saved:", f"{save_dir}/sample.obj")


if __name__ == "__main__":
    sample()
