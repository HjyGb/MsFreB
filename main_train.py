# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
"""
Main training script for the Image Manipulation Localization (IML) model.

This script handles the entire training and evaluation pipeline, including:
- Parsing command-line arguments.
- Setting up distributed training.
- Creating datasets and data loaders.
- Initializing the model (Msfreb), optimizer, and loss scaler.
- Running the training and evaluation loops.
- Logging results to TensorBoard and a log file.
- Saving model checkpoints.
"""
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm.optim as optim_factory

import utils.datasets
import utils.iml_transforms
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler


import msfreb

from engine_train import train_one_epoch, test_one_epoch

def get_args_parser():
    """
    Parses and returns command-line arguments for training.
    """
    parser = argparse.ArgumentParser('IML-ViT training', add_help=True)
    
    # --- Batch and Epoch Settings ---
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
    parser.add_argument('--test_batch_size', default=2, type=int,
                        help="Batch size for testing.")
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--test_period', default=4, type=int,
                        help="Run testing every N epochs.")
    parser.add_argument('--accum_iter', default=16, type=int,
                        help='Accumulate gradient iterations to increase effective batch size.')

    # --- Model Backbone Configuration ---
    parser.add_argument('--convnext_variant', default='base', type=str, choices=['tiny', 'base', 'large'],
                        help='Specify the ConvNeXt variant: "tiny", "base", or "large".')
    parser.add_argument('--convnext_pretrain_path', default=None, type=str, 
                        help='Path to pretrained weights for ConvNeXt.')

    # --- Loss Function Hyperparameters ---
    parser.add_argument('--edge_broaden', default=7, type=int,
                        help='Width (in pixels) to broaden the edge mask for edge loss calculation.')
    parser.add_argument('--edge_lambda', default=20, type=float,
                        help='Weight for the edge loss component.')
    parser.add_argument('--focal_alpha', default=0.25, type=float,
                        help='Alpha parameter for Focal Loss.')
    parser.add_argument('--focal_gamma', default=2.0, type=float,
                        help='Gamma parameter for Focal Loss.')
    parser.add_argument('--dice_smooth', default=1.0, type=float,
                        help='Smoothing factor for Dice Loss to prevent division by zero.')
    parser.add_argument('--dice_weight', default=1.0, type=float,
                        help='Weight for the Dice Loss component.')
    parser.add_argument('--classification_loss_weight', default=1.0, type=float,
                        help='Weight for the image-level classification loss.')
    parser.add_argument('--consistency_loss_weight', default=0.5, type=float,
                        help='Weight for the consistency loss between segmentation and classification predictions.')

    # --- Loss Penalty Hyperparameters ---
    parser.add_argument('--seg_penalty_fp', default=1.5, type=float,
                        help='Penalty factor for segmentation loss on false positives (predicting tamper on authentic images).')
    parser.add_argument('--seg_penalty_fn', default=1.5, type=float,
                        help='Penalty factor for segmentation loss on false negatives (failing to predict tamper on manipulated images).')

    # --- Model Architecture Configuration ---
    parser.add_argument('--clf_feature_backbone_index', default=2, type=int,
                        help='Index of the backbone feature map to use for the classification head (e.g., 0 for C2, 2 for C4).')
    parser.add_argument('--ppm_norm_type', default='gn', type=str, choices=['bn', 'gn', 'ln'],
                        help='Normalization layer for PPM. "gn" (GroupNorm) is recommended for small batch sizes.')
    parser.add_argument('--predict_head_norm', default="LN", type=str, choices=['BN', 'LN', 'IN'],
                        help="Normalization for the prediction head ('BN', 'LN', 'IN').")
    
    # Note: Architecture parameters (use_ppm, use_wavelet_enhancement, bifpn_attention, bifpn_use_p8)
    # are not exposed as they use model defaults for consistency

    # --- Optimizer Parameters ---
    parser.add_argument('--optimizer_name', default='AdamW', type=str, choices=['AdamW', 'SGD'],
                        help='Optimizer to use.')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Gradient clipping norm (e.g., 1.0). Default is None (no clipping).')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: 0.05).')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute value). If None, calculated from base_lr.')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='Base learning rate. Absolute LR = base_lr * total_batch_size / 256.')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Lower bound for learning rate in cyclic schedulers.')
    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N',
                        help='Number of epochs for learning rate warmup.')
    
    # --- Dataset and Path Parameters ---
    parser.add_argument('--data_path', default='/root/Dataset/CASIA2.0/', type=str,
                        help='Path to the training dataset (directory or JSON file).')
    parser.add_argument('--test_data_path', default='/root/Dataset/CASIA1.0', type=str,
                        help='Path to the testing dataset (directory or JSON file).')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Path to save checkpoints and logs.')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='Path for TensorBoard logs.')
    parser.add_argument('--resume', default='',
                        help='Path to a checkpoint file to resume training from.')

    # --- Environment and Reproducibility ---
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing (e.g., "cuda" or "cpu").')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch number.')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfer.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # --- Distributed Training Parameters ---
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes.')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training.')
    return parser

def main(args):
    # Initialize distributed mode
    misc.init_distributed_mode(args)
    # Set multiprocessing sharing strategy to prevent file descriptor errors
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    print('Job directory: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seed for reproducibility
    seed = args.seed + misc.get_rank()
    misc.seed_torch(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get data augmentation pipelines
    train_transform = utils.iml_transforms.get_albu_transforms('train')
    test_transform = utils.iml_transforms.get_albu_transforms('test')

    # --- Dataset Loading ---
    # Handle both directory-based and JSON-based datasets
    if os.path.isdir(args.data_path):
        dataset_train = utils.datasets.mani_dataset(args.data_path, transform=train_transform, edge_width=args.edge_broaden, if_return_shape=True)
    else:
        dataset_train = utils.datasets.json_dataset(args.data_path,transform=train_transform, edge_width = args.edge_broaden, if_return_shape = True)
    
    if os.path.isdir(args.test_data_path):
        dataset_test = utils.datasets.mani_dataset(args.test_data_path, transform=test_transform, edge_width=args.edge_broaden, if_return_shape=True)
    else:
        dataset_test = utils.datasets.json_dataset(args.test_data_path,transform=test_transform, edge_width = args.edge_broaden, if_return_shape = True)

    print(f"Training dataset size: {len(dataset_train)}")
    print(f"Testing dataset size: {len(dataset_test)}")

    # --- Data Sampler for Distributed Training ---
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False # No need to shuffle test set
        )
        print(f"Sampler_train: {sampler_train}")
        print(f"Sampler_test: {sampler_test}")
    else:
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test) # Sequential for testing

    # --- TensorBoard Logger ---
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # --- Data Loaders ---
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    # --- Model Definition ---
    model = msfreb.Msfreb(
        # General parameters
        predict_head_norm=args.predict_head_norm,
        edge_lambda=args.edge_lambda,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        dice_smooth=args.dice_smooth,
        dice_weight=args.dice_weight,
        classification_loss_weight=args.classification_loss_weight,
        seg_penalty_fp=args.seg_penalty_fp,
        seg_penalty_fn=args.seg_penalty_fn,
        clf_feature_backbone_index=args.clf_feature_backbone_index,
        consistency_loss_weight=args.consistency_loss_weight,
        # PPM normalization type (architecture params use defaults)
        ppm_norm_type=args.ppm_norm_type,
        # Backbone parameters
        backbone_type='convnext',
        convnext_variant=args.convnext_variant,
        convnext_pretrain_path=args.convnext_pretrain_path
        # Note: use_ppm=True, use_wavelet_enhancement=True, bifpn_attention=True (defaults)
    )
    
    if args.distributed:
        # SyncBatchNorm is crucial for consistent BN stats in DDP
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)
    model_without_ddp = model
    print(f"Model:\n{str(model_without_ddp)}")

    # --- Learning Rate and Batch Size Calculation ---
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # Calculate LR if not specified
        args.lr = args.blr * eff_batch_size / 256

    print(f"Base LR: {args.blr:.2e}")
    print(f"Actual LR: {args.lr:.2e}")
    print(f"Accumulate grad iterations: {args.accum_iter}")
    print(f"Effective batch size: {eff_batch_size}")

    # --- DDP Wrapper ---
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    # --- Optimizer and Loss Scaler ---
    # Set weight decay to 0 for bias and normalization layers for better stability
    args.opt = args.optimizer_name 
    args.betas=(0.9, 0.999)
    args.momentum=0.9
    optimizer = optim_factory.create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    print(f"Using optimizer: {args.optimizer_name}\n{optimizer}")

    # --- Load Checkpoint if Resuming ---
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # --- Training Loop ---
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_f1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        
        # --- Save Checkpoint Periodically ---
        if args.output_dir and (epoch % 10 == 0 and epoch != 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            
        optimizer.zero_grad()

        # --- Evaluation and Best Model Saving ---
        if epoch % args.test_period == 0 or epoch + 1 == args.epochs:
            test_stats = test_one_epoch(
                model, 
                data_loader=data_loader_test, 
                device=device, 
                epoch=epoch, 
                log_writer=log_writer,
                args=args
            )
            current_f1 = test_stats['average_f1']
            if current_f1 > best_f1:
                best_f1 = current_f1
                print(f"*** New best F1 score: {best_f1:.4f} ***")
                # Save the best model, especially after an initial warmup/stabilization period
                if epoch > 35 and args.output_dir:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, is_best=True)
                
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        
        # --- Log to File ---
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
