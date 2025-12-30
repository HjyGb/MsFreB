# MsFreB: Multi-scale Frequency and Bi-supervision for Manipulation Detection

![Powered by](https://img.shields.io/badge/Based_on-Pytorch-blue?logo=pytorch) 

This repository contains the implementation of MsFreB for image manipulation detection and localization.


## 1 Environment Requirements

Ubuntu 22.04

Python 3.13.5

PyTorch 2.7.1

## 2 Training

### 2.1 Dataset Preparation

The dataset structure follows the same requirements as the original implementation. We provide two utility tools to help prepare your datasets:

#### Image Resolution Tool
For datasets with overly large image resolutions, you can use `tools/resize.py` to resize all images to a desired maximum size:

1. Copy `tools/resize.py` to the root directory of this repository
2. Configure the paths in the script according to your dataset structure
3. Run the script to resize images while maintaining aspect ratio

#### Dataset Index Generation
Use `tools/image_process.py` to generate JSON index files for your dataset:

1. Copy `tools/image_process.py` to the root directory of this repository  
2. Configure the paths in the script to point to your dataset directories
3. Run the script to generate the JSON index file

Both tools need to be placed in the repository root directory and have their paths properly configured before use.

### 2.2 Pre-trained Weights

Download the ConvNeXt-V2 pre-trained weights from the [official repository](https://github.com/facebookresearch/ConvNeXt-V2). We use the base model available at:
- https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt

Thanks to the ConvNeXt-V2 team for their excellent work and open-source contributions.

### 2.3 Training Script

Use the provided training script with the following modifications:

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
main_train.py \
  --world_size 1 \
  --batch_size 2 \
  --data_path "./path/to/your/dataset.json" \
  --epochs 200 \
  --lr 1e-5 \
  --min_lr 5e-7 \
  --weight_decay 0.05 \
  --edge_lambda 20 \
  --focal_alpha 0.1 \
  --convnext_variant base \
  --convnext_pretrain_path "./path/to/convnextv2_base_1k_224_ema.pth" \
  --test_data_path "./path/to/your/test_dataset.json" \
  --warmup_epochs 4 \
  --output_dir ./output/ \
  --log_dir ./output/  \
  --accum_iter 8 \
  --seed 42 \
  --test_period 4 \
  --num_workers 4 \
  2> train_error.log 1>train_log.log
```

**Key Parameters:**
- `data_path` and `test_data_path`: Can be either dataset directories or JSON file paths
- For multi-GPU training: Set `nproc_per_node` and `world_size` to the number of GPUs
- You can reduce `accum_iter` accordingly to maintain the same total batch size
  - Total effective batch size = `batch_size` × `world_size` × `accum_iter`
- All outputs are redirected to log files for monitoring

## 3 Testing

We provide a comprehensive testing script that handles both performance benchmarking and accuracy evaluation. 

To use the testing script:

1. Configure the following paths in `test.sh`:
   - `model_path`: Path to your trained model checkpoint
   - `data_path`: Path to your test dataset
   - `output_dir`: Directory to save test results

2. Run the test script:
   ```bash
   ./test.sh
   ```


