# 3D-Diffusion-final-project

This project trains a PointNet++ U-Net DDPM for ShapeNet Airplane point cloud generation.

Assume the current terminal is in the parent directory of this project:

```bash
cd 3D-Diffusion-final-project
conda activate final
```

## Paths

```text
Raw ShapeNet airplane meshes: data/02691156/
Preprocessed point clouds:    data/airplane_points/
Checkpoint:                   checkpoints/pointnet_unet_airplane.pth
Generated point cloud:         outputs/airplane_sample.ply
```

## Prepare Data

If `data/airplane_points/` already contains `.npy` point clouds, skip this step.

`preprocess.py` reads airplane meshes from:

```text
data/02691156/
```

and saves point clouds to:

```text
data/airplane_points/
```

Run:

```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python preprocess.py
```

Each point cloud has shape:

```text
[16384, 3]
```

## Train

```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python train.py
```

The checkpoint is saved to:

```text
checkpoints/pointnet_unet_airplane.pth
```

## Sample

```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python sample.py
```

The generated point cloud is saved to:

```text
outputs/airplane_sample.ply
```