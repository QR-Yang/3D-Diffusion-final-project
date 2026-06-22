# Point Cloud DDPM

This project trains a DDPM point cloud generator.

The training data is saved as `.npy` point clouds:

```text
data/airplane_points/
```

Each sample has shape:

```text
[16384, 3]
```

## Prepare Data

If the `.npy` point clouds already exist, skip this step.

To convert ShapeNet obj files to point clouds:

```bash
cd unet_pointnet++_pointcloud
conda activate final
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python preprocess.py
```

Current `preprocess.py` uses:

```text
input_dir  = data/03001627
output_dir = data/chair_points
```

Current `data.py` reads:

```text
data/airplane_points
```

Before training, make sure `preprocess.py` output path and `data.py` input path are the same.

## Train

```bash
cd unet_pointnet++_pointcloud
conda activate final
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python train.py
```

The checkpoint is saved to:

```text
checkpoints/pointnet_unet_airplain_8192*2_2e-5_8000.pth
```

## Sample

Before sampling, make sure `checkpoint_path` in `sample.py` matches the checkpoint produced by `train.py`.

Then run:

```bash
cd unet_pointnet++_pointcloud
conda activate final
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python sample.py
```

The generated point cloud is saved to:

```text
outputs/sample_chair.ply
```
