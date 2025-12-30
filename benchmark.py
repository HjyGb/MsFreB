#!/usr/bin/env python3
"""
Model inference performance benchmark script.
Tests inference speed and GPU memory usage with randomly initialized model weights.
Focuses purely on performance metrics without loading pretrained weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import os
import gc
import psutil
import numpy as np
from contextlib import contextmanager
from typing import Tuple, List, Dict, Any

# Import model
from msfreb import Msfreb

def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information in MB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        return {
            'allocated': allocated,
            'reserved': reserved, 
            'max_allocated': max_allocated
        }
    return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}

def get_cpu_memory_info() -> float:
    """Get CPU memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

@contextmanager
def measure_memory():
    """Memory measurement context manager"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    cpu_before = get_cpu_memory_info()
    gpu_before = get_gpu_memory_info()
    
    yield
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    cpu_after = get_cpu_memory_info()
    gpu_after = get_gpu_memory_info()
    
    print(f"CPU Memory Usage: {cpu_after - cpu_before:.2f} MB")
    print(f"GPU Memory Allocated: {gpu_after['allocated']:.2f} MB")
    print(f"GPU Memory Reserved: {gpu_after['reserved']:.2f} MB") 
    print(f"GPU Max Memory Allocated: {gpu_after['max_allocated']:.2f} MB")

def create_dummy_data(batch_size: int, input_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create dummy input data for benchmarking"""
    # Input images (B, 3, H, W)
    images = torch.randn(batch_size, 3, input_size, input_size, device=device, dtype=torch.float32)
    
    # Dummy masks (B, 1, H, W) - not used during inference but required by model
    masks = torch.zeros(batch_size, 1, input_size, input_size, device=device, dtype=torch.float32)
    
    # Dummy edge masks (B, 1, H, W) - not used during inference but required by model 
    edge_masks = torch.zeros(batch_size, 1, input_size, input_size, device=device, dtype=torch.float32)
    
    return images, masks, edge_masks

def warmup_model(model: nn.Module, device: str, input_size: int, warmup_steps: int = 10):
    """Warmup model for stable benchmarking"""
    print(f"Warming up model... ({warmup_steps} steps)")
    model.eval()
    
    with torch.no_grad():
        dummy_input, dummy_masks, dummy_edge_masks = create_dummy_data(1, input_size, device)
        
        for _ in range(warmup_steps):
            # Model forward pass - only get prediction results, ignore loss-related outputs
            outputs = model(dummy_input, dummy_masks, dummy_edge_masks, shape=None)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    print("Warmup completed")

def benchmark_inference(
    model: nn.Module, 
    batch_size: int,
    input_size: int, 
    device: str,
    num_iterations: int = 100,
    warmup_steps: int = 10
) -> Dict[str, Any]:
    """Benchmark inference performance"""
    
    print(f"\n{'='*60}")
    print(f"Inference Performance Benchmark")
    print(f"Batch Size: {batch_size}")
    print(f"Input Size: {input_size}x{input_size}")
    print(f"Device: {device}")
    print(f"Iterations: {num_iterations}")
    print(f"{'='*60}")
    
    # Warmup
    warmup_model(model, device, input_size, warmup_steps)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    model.eval()
    times = []
    
    # Create test data
    test_images, test_masks, test_edge_masks = create_dummy_data(batch_size, input_size, device)
    
    print(f"\nStarting inference benchmark...")
    
    with torch.no_grad():
        with measure_memory():
            for i in range(num_iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                
                # Inference
                outputs = model(test_images, test_masks, test_edge_masks, shape=None)
                # Main outputs of interest:
                # outputs[0]: total_loss (used during training)
                # outputs[1]: mask_pred_prob (segmentation prediction probability map)
                # outputs[2]: tamper_pred_prob (image-level tampering probability)
                mask_pred = outputs[1]  # Segmentation prediction results
                tamper_pred = outputs[2]  # Image-level prediction results
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                inference_time = end_time - start_time
                times.append(inference_time)
                
                if (i + 1) % 20 == 0:
                    avg_time = np.mean(times[-20:])
                    fps = batch_size / avg_time
                    print(f"Iteration {i+1:3d}/{num_iterations}: {inference_time*1000:.2f}ms, "
                          f"Recent 20 avg: {avg_time*1000:.2f}ms, FPS: {fps:.2f}")
    
    # Statistical results
    times = np.array(times)
    avg_time = np.mean(times) 
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = batch_size / avg_time
    throughput = batch_size * len(times) / np.sum(times)
    
    results = {
        'batch_size': batch_size,
        'input_size': input_size,
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000, 
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'fps': fps,
        'throughput': throughput,
        'gpu_memory': get_gpu_memory_info()
    }
    
    return results

def print_results(results: Dict[str, Any]):
    """Print benchmark results"""
    print(f"\n{'='*60}")
    print(f"Inference Performance Results")
    print(f"{'='*60}")
    print(f"Batch Size: {results['batch_size']}")
    print(f"Input Size: {results['input_size']}x{results['input_size']}")
    print(f"Average Time: {results['avg_time_ms']:.2f} ± {results['std_time_ms']:.2f} ms")
    print(f"Min Time: {results['min_time_ms']:.2f} ms")
    print(f"Max Time: {results['max_time_ms']:.2f} ms") 
    print(f"FPS: {results['fps']:.2f}")
    print(f"Throughput: {results['throughput']:.2f} images/sec")
    
    gpu_mem = results['gpu_memory']
    if gpu_mem['allocated'] > 0:
        print(f"GPU Memory Allocated: {gpu_mem['allocated']:.2f} MB")
        print(f"GPU Memory Reserved: {gpu_mem['reserved']:.2f} MB")
        print(f"GPU Max Memory: {gpu_mem['max_allocated']:.2f} MB")
    print(f"{'='*60}")

def test_multiple_batch_sizes(
    model: nn.Module,
    batch_sizes: List[int],
    input_size: int,
    device: str,
    num_iterations: int = 50
):
    """Test performance with multiple batch sizes"""
    all_results = []
    
    for batch_size in batch_sizes:
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            results = benchmark_inference(
                model, batch_size, input_size, device, num_iterations
            )
            all_results.append(results)
            print_results(results)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\nError: Batch size {batch_size} out of memory: {str(e)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            else:
                raise e
    
    # Summary results
    if all_results:
        print(f"\n{'='*80}")
        print(f"All Benchmark Results Summary")
        print(f"{'='*80}")
        print(f"{'Batch':<8}{'Avg Time(ms)':<15}{'FPS':<10}{'Throughput':<12}{'GPU Mem(MB)':<12}")
        print(f"{'-'*80}")
        
        for result in all_results:
            gpu_mem = result['gpu_memory']['max_allocated']
            print(f"{result['batch_size']:<8}"
                  f"{result['avg_time_ms']:<15.2f}"
                  f"{result['fps']:<10.2f}"
                  f"{result['throughput']:<12.2f}"
                  f"{gpu_mem:<12.2f}")
        print(f"{'='*80}")

def create_model(args) -> nn.Module:
    """Create model with random initialization (no pretrained weights)"""
    print("Creating model with random initialization...")
    
    model = Msfreb(
        backbone_type='convnext',
        convnext_variant=args.convnext_variant,
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
        ppm_norm_type=args.ppm_norm_type
        # Note: Architecture parameters use model defaults:
        # - bifpn_attention=True, bifpn_use_p8=False  
        # - use_ppm=True, use_wavelet_enhancement=True
    )
    
    print("Model created with random weights for benchmarking")
    return model

def get_args_parser():
    parser = argparse.ArgumentParser('Model Inference Performance Benchmark', add_help=True)
    
    # Basic parameters
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Computing device')
    parser.add_argument('--input_size', default=1024, type=int,
                        help='Input image size')  
    parser.add_argument('--batch_sizes', default=[1, 2, 4, 8], nargs='+', type=int,
                        help='Batch sizes to test')
    parser.add_argument('--num_iterations', default=100, type=int,
                        help='Number of iterations for each batch size')
    parser.add_argument('--warmup_steps', default=10, type=int,
                        help='Number of warmup steps')
    
    # Model parameters
    parser.add_argument('--convnext_variant', default='base', choices=['tiny', 'base', 'large'],
                        help='ConvNeXt variant')
    parser.add_argument('--fpn_channels', default=256, type=int,
                        help='FPN channels')
    parser.add_argument('--mlp_embed_dim', default=256, type=int,
                        help='MLP embedding dimension')
    parser.add_argument('--predict_head_norm', default='LN', choices=['BN', 'LN', 'IN'],
                        help='Prediction head normalization type')
    parser.add_argument('--dropout_rate', default=0.1, type=float,
                        help='Dropout rate')
    
    # Loss-related parameters (required for model initialization)
    parser.add_argument('--edge_lambda', default=20, type=float,
                        help='Edge loss weight')
    parser.add_argument('--focal_alpha', default=0.25, type=float,
                        help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', default=2.0, type=float,
                        help='Focal loss gamma parameter')
    parser.add_argument('--dice_smooth', default=1.0, type=float,
                        help='Dice loss smoothing factor')
    parser.add_argument('--dice_weight', default=1.0, type=float,
                        help='Dice loss weight')
    parser.add_argument('--classification_loss_weight', default=1.0, type=float,
                        help='Classification loss weight')
    parser.add_argument('--seg_penalty_fp', default=1.5, type=float,
                        help='Segmentation FP penalty factor')
    parser.add_argument('--seg_penalty_fn', default=1.5, type=float,
                        help='Segmentation FN penalty factor')
    parser.add_argument('--clf_feature_backbone_index', default=2, type=int,
                        help='Backbone layer index for classification features')
    parser.add_argument('--consistency_loss_weight', default=0.5, type=float,
                        help='Consistency loss weight')
    parser.add_argument('--ppm_norm_type', default='gn', choices=['bn', 'gn', 'ln'],
                        help='PPM normalization type')
    
    # Note: Architecture parameters (bifpn_attention, bifpn_use_p8, use_ppm, 
    # use_wavelet_enhancement, etc.) are not exposed as they use model defaults
    
    return parser

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create model with random initialization
    model = create_model(args)
    model = model.to(args.device)
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} ({trainable_params:,} trainable)")
    
    # Start performance benchmarking
    test_multiple_batch_sizes(
        model=model,
        batch_sizes=args.batch_sizes,
        input_size=args.input_size,
        device=args.device,
        num_iterations=args.num_iterations
    )

if __name__ == '__main__':
    main()
