# Visual-Selective-VIO (ECCV 2022)

This repository contains the codes for [Efficient Deep Visual and Inertial Odometry with Adaptive Visual Modality Selection (ECCV '22)](https://arxiv.org/pdf/2205.06187.pdf). 

<img src="figures/figure.png" alt="overview" width="700"/> 

## Overview

This work proposes an adaptive deep-learning based Visual-Inertial Odometry (VIO) method that reduces computational redundancy by **opportunistically disabling the visual modality**. A policy network learns to deactivate the visual feature extractor on-the-fly based on the current motion state and IMU readings, using the Gumbel-Softmax trick for end-to-end differentiable training.

**Key results:** Similar or better accuracy than full-modality baselines with **up to 78.8% computational complexity reduction** on KITTI.

## Requirements

- Python 3.8+
- PyTorch >= 1.7 (tested with 2.6.0+cu124)
- scipy
- tqdm
- path (`pip install path`)
- matplotlib
- numpy
- PIL (Pillow)

Install all dependencies:
```bash
pip install torch torchvision scipy tqdm path matplotlib numpy Pillow
```

## Data Preparation

The code is tested on the **KITTI Odometry** dataset. The project expects the following data structure under `data/kitti_data/`:

```
data/kitti_data/
├── imus/          # Pre-processed IMU data (.mat files)
│   ├── 00.mat
│   ├── 01.mat
│   └── ...
├── poses/         # Ground truth poses (.txt files)
│   ├── 00.txt
│   ├── 01.txt
│   └── ...
└── sequences/     # KITTI image sequences
    ├── 00/
    │   └── image_2/   # Left camera images (.png)
    ├── 01/
    └── ...
```

The IMU data (pre-processed) is provided under `data/kitti_data/imus/`. To download the images and poses, run:

```bash
cd data
source data_prep.sh 
```

> **Note:** Sequence 03 does not have IMU data and is excluded from both training and validation splits.

## IMU Data Format

The IMU data has 6 dimensions: 

1. acceleration in x, i.e. in direction of vehicle front (m/s^2)
2. acceleration in y, i.e. in direction of vehicle left (m/s^2)
3. acceleration in z, i.e. in direction of vehicle top (m/s^2)
4. angular rate around x (rad/s)
5. angular rate around y (rad/s)
6. angular rate around z (rad/s)

## Download Pretrained Models

We provide two pretrained checkpoints `vf_512_if_256_3e-05.model` and `vf_512_if_256_5e-05.model` and two pretrained FlowNet models in [Link](https://drive.google.com/drive/folders/1KrxpvUV9Bn5SwUlrDKe76T2dqF1ooZyk). Please download them and place them under `pretrain_models` directory.

## Test the Pretrained Model

Run evaluation on the validation sequences (05, 07, 10):

```bash
python test.py --data_dir './data/kitti_data' --model './pretrain_models/vf_512_if_256_5e-05.model' --gpu_ids '0' --experiment_name 'pretrained'
```

The figures and error records will be generated under `./results/pretrained/files/`.

### Test Results (vf_512_if_256_5e-05.model)

| Sequence | t_rel (%) | r_rel (deg/100m) | t_rmse (m) | r_rmse (deg) | Usage (%) |
|----------|-----------|-------------------|------------|---------------|-----------|
| 05       | 2.4327    | 0.9154            | 0.0347     | 0.0501        | 11.45     |
| 07       | 2.1361    | 1.1947            | 0.0401     | 0.0478        | 10.45     |
| 10       | 3.4037    | 1.1762            | 0.0655     | 0.0590        | 12.50     |

The estimated path (left), decision heatmap (middle) and speed heatmap (right) for path 07:

<img src="figures/07_path_2d.png" alt="path" height="230"/> <img src="figures/07_decision_smoothed.png" alt="path" height="230"/> <img src="figures/07_speed.png" alt="path" height="230"/>

## Train the Model

The training has three stages:
1. **Warmup** (40 epochs): Train pose network with random visual modality selection
2. **Joint** (40 epochs): Train pose + policy network together with Gumbel-Softmax
3. **Fine-tuning** (20 epochs): Fine-tune with low learning rate

```bash
python train.py --data_dir './data/kitti_data' --gpu_ids '0' --experiment_name 'my_experiment' --batch_size 16
```

### Key Training Arguments

| Argument            | Default      | Description                                      |
|---------------------|-------------|--------------------------------------------------|
| `--data_dir`        | `./data/kitti_data` | Path to the KITTI dataset                |
| `--batch_size`      | 16          | Batch size (reduce for limited VRAM)              |
| `--epochs_warmup`   | 40          | Warmup stage epochs                               |
| `--epochs_joint`    | 40          | Joint training epochs                             |
| `--epochs_fine`     | 20          | Fine-tuning epochs                                |
| `--fuse_method`     | `cat`       | Fusion method: `cat`, `soft`, `hard`              |
| `--Lambda`          | 3e-5        | Penalty factor for visual encoder usage           |
| `--pretrain_flownet`| `./pretrain_models/flownets_bn_EPE2.459.pth.tar` | Pre-trained FlowNet weights |

## Project Structure

```
Visual-Selective-VIO/
├── model.py                    # DeepVIO model (Encoder, PolicyNet, Pose_RNN, Fusion)
├── train.py                    # Three-stage training script
├── test.py                     # Evaluation script
├── dataset/
│   └── KITTI_dataset.py        # KITTI dataset loader with LDS weighting
├── utils/
│   ├── utils.py                # Pose utilities, KITTI error metrics
│   ├── kitti_eval.py           # KITTI evaluation & plotting
│   └── custom_transform.py     # Data augmentation transforms
├── data/
│   ├── data_prep.sh            # Script to download KITTI data
│   └── kitti_data/             # KITTI dataset (not tracked in git)
├── pretrain_models/            # Pretrained model checkpoints (not tracked in git)
└── figures/                    # Example output figures
```

## Reference

> Mingyu Yang, Yu Chen, Hun-Seok Kim, "Efficient Deep Visual and Inertial Odometry with Adaptive Visual Modality Selection"

    @article{yang2022efficient,
      title={Efficient Deep Visual and Inertial Odometry with Adaptive Visual Modality Selection},
      author={Yang, Mingyu and Chen, Yu and Kim, Hun-Seok},
      journal={arXiv preprint arXiv:2205.06187},
      year={2022}
    }
