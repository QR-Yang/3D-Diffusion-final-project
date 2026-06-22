from pathlib import Path
import numpy as np
import trimesh
input_dir = Path("data/02691156")
output_dir = Path("data/airplane_points")
num_points = 8192 * 2
def preprocess():
    output_dir.mkdir(exist_ok=True)
    files = sorted(input_dir.glob("*/model.obj"))
    for file in files:
        name = file.parent.name
        save_path = output_dir / f"{name}.npy"
        mesh = trimesh.load(file, force="mesh")
        points = mesh.sample(num_points).astype("float32")
        points = points - points.mean(axis=0)
        scale = np.sqrt((points ** 2).sum(axis=1)).max()
        points = points / scale
        np.save(save_path, points)
if __name__ == "__main__":
    preprocess()
