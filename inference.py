#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model inference accuracy evaluation tool.
Focuses on accuracy metrics evaluation without performance measurements.
Follows test_one_epoch logic for consistent accuracy assessment.
"""

import argparse
import os
import json
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, jaccard_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import custom modules
import utils.datasets as datasets
import utils.iml_transforms as iml_transforms
import utils.evaluation as evaluation
import utils.misc as misc
from msfreb import Msfreb


def get_args_parser():
    parser = argparse.ArgumentParser('Model Inference Accuracy Test', add_help=True)
    
    # Model parameters
    parser.add_argument('--model_path', required=True, type=str, help='Path to trained model weights file')
    parser.add_argument('--convnext_variant', default='base', type=str, choices=['tiny', 'base', 'large'], help='ConvNeXt variant')
    
    # Data parameters
    parser.add_argument('--data_path', required=True, type=str, help='Test dataset path')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers')
    
    # Output parameters
    parser.add_argument('--output_dir', required=True, type=str, help='Results save directory')
    parser.add_argument('--save_predictions', action='store_true', help='Whether to save prediction result images')
    
    # Model configuration parameters
    parser.add_argument('--edge_broaden', default=7, type=int, help='Edge broadening size')
    parser.add_argument('--edge_lambda', default=20, type=float, help='Edge loss weight')
    parser.add_argument('--focal_alpha', default=0.25, type=float, help='Focal Loss alpha parameter')
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='Focal Loss gamma parameter')
    parser.add_argument('--dice_smooth', default=1.0, type=float, help='Dice Loss smoothing factor')
    parser.add_argument('--dice_weight', default=1.0, type=float, help='Dice Loss weight')
    parser.add_argument('--classification_loss_weight', default=1.0, type=float, help='Classification loss weight')
    parser.add_argument('--seg_penalty_fp', default=1.5, type=float, help='Segmentation false positive penalty factor')
    parser.add_argument('--seg_penalty_fn', default=1.5, type=float, help='Segmentation false negative penalty factor')
    parser.add_argument('--clf_feature_backbone_index', default=2, type=int, help='Backbone feature index for classification head')
    parser.add_argument('--consistency_loss_weight', default=0.5, type=float, help='Consistency loss weight')
    parser.add_argument('--ppm_norm_type', default='gn', type=str, choices=['bn', 'gn', 'ln'], help='PPM normalization type')
    parser.add_argument('--predict_head_norm', default="LN", type=str, help='Prediction head normalization type')
    
    # Note: Architecture parameters (use_ppm, use_wavelet_enhancement, bifpn_attention, etc.)
    # are not exposed as command line arguments - they use model defaults
    
    # Device parameters
    
    # Device parameters
    parser.add_argument('--device', default='cuda', type=str, help='Inference device')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    
    return parser


def save_prediction_image(image, mask_pred, mask_gt, tamper_pred, filename, output_dir):
    """Save single prediction result image"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process image
    if isinstance(image, torch.Tensor):
        image = image.cpu()
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = mask_pred.cpu()
    if isinstance(mask_gt, torch.Tensor):
        mask_gt = mask_gt.cpu()
    
    # Denormalize
    image_denorm = datasets.denormalize(image.unsqueeze(0))[0]  # 1,3,H,W -> 3,H,W
    
    # Convert to numpy and adjust dimensions
    image_np = image_denorm.permute(1, 2, 0).numpy()  # 3,H,W -> H,W,3
    image_np = np.clip(image_np, 0, 1)
    image_np = (image_np * 255).astype(np.uint8)
    
    # Process masks
    if len(mask_pred.shape) > 2:
        mask_pred = mask_pred[0]  # Remove channel dimension
    if len(mask_gt.shape) > 2:
        mask_gt = mask_gt[0]
    
    mask_pred_np = (mask_pred.numpy() * 255).astype(np.uint8)
    mask_gt_np = (mask_gt.numpy() * 255).astype(np.uint8)
    
    # Create combined image
    h, w = image_np.shape[:2]
    combined = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    
    # Place images
    combined[:h, :w] = image_np
    combined[:h, w:] = np.stack([mask_pred_np, mask_pred_np, mask_pred_np], axis=2)
    combined[h:, :w] = np.stack([mask_gt_np, mask_gt_np, mask_gt_np], axis=2)
    
    # Add classification text
    tamper_result = "true" if tamper_pred else "false"
    cv2.putText(combined, f"Tamper: {tamper_result}", (w + 10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save
    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}_{tamper_result}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


def run_inference(args):
    """Run inference test - follows test_one_epoch logic exactly"""
    # Set device and random seed
    device = torch.device(args.device)
    misc.seed_torch(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset - same way as during training
    test_transform = iml_transforms.get_albu_transforms('test')
    
    if os.path.isdir(args.data_path):
        dataset_test = datasets.mani_dataset(
            args.data_path, 
            transform=test_transform, 
            edge_width=args.edge_broaden,
            if_return_shape=True,
            if_return_name=True
        )
    else:
        dataset_test = datasets.json_dataset(
            args.data_path,
            transform=test_transform, 
            edge_width=args.edge_broaden,
            if_return_shape=True,
            if_return_name=True
        )
    
    print(f"Test dataset size: {len(dataset_test)}")
    
    # Create data loader - same as during training
    data_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Create model
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
        # - use_ppm=True, use_wavelet_enhancement=True, bifpn_attention=True, etc.
    )
    
    # Load weights
    try:
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=True)
    except:
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully: {args.model_path}")
    
    # Create prediction results save directory
    prediction_dir = os.path.join(args.output_dir, 'predictions')
    
    # Statistics variables
    total_images = 0
    
    # Initialize metric_logger - same as test_one_epoch
    metric_logger = misc.MetricLogger(delimiter="  ")
    
    # Same logic as test_one_epoch
    all_tamper_preds = []
    all_tamper_gts = []
    
    # Cumulative variables for online segmentation metric calculation
    seg_tp_total = 0
    seg_tn_total = 0
    seg_fp_total = 0
    seg_fn_total = 0
    seg_auc_scores = []  # Only store AUC scores for each batch, not raw data
    
    print("Starting inference...")
    
    with torch.no_grad():
        model.zero_grad()  # Same as test_one_epoch
        
        print_freq = 20
        header = 'Inference'
        
        for data_iter_step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            # Unpack data - same order as during training
            images = batch_data[0]  # image
            masks = batch_data[1]   # mask  
            edge_masks = batch_data[2]  # edge_mask
            
            # Get additional information
            filenames = None
            shapes = None
            
            if len(batch_data) > 3:
                # Determine if 4th element is filename or shape
                if isinstance(batch_data[3], list) or (hasattr(batch_data[3], 'dtype') and 'str' in str(batch_data[3].dtype)):
                    filenames = batch_data[3]
                    if len(batch_data) > 4:
                        shapes = batch_data[4]
                else:
                    shapes = batch_data[3]
                    if len(batch_data) > 4:
                        filenames = batch_data[4]
            
            if shapes is None:
                shapes = torch.tensor([[1024, 1024] for _ in range(images.shape[0])], dtype=torch.long)
            if filenames is None:
                filenames = [f"img_{total_images + i}" for i in range(images.shape[0])]
            
            # Move to device - same as test_one_epoch
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            edge_masks = edge_masks.to(device, non_blocking=True)
            
            # Model inference - exactly same call as test_one_epoch
            outputs = model(images, masks, edge_masks, shapes)
            predict_mask_prob = outputs[1]
            tamper_prob = outputs[2]
            
            # Same processing as test_one_epoch
            predict_mask_prob = predict_mask_prob.detach()
            tamper_prob = tamper_prob.detach()
            
            # Calculate image-level GT - same as test_one_epoch
            is_tampered_gt_batch = (masks.sum(dim=[1, 2, 3]) > 1e-5).float().unsqueeze(1)
            all_tamper_preds.append(tamper_prob.cpu())
            all_tamper_gts.append(is_tampered_gt_batch.cpu())
            
            # Segmentation evaluation - same as test_one_epoch
            region_mask = evaluation.genertate_region_mask(masks, shapes)
            TP, TN, FP, FN = evaluation.cal_confusion_matrix(predict_mask_prob, masks, region_mask)
            local_f1 = evaluation.cal_F1(TP, TN, FP, FN)
            
            for i in local_f1:
                metric_logger.update(average_f1=i)
            
            # Online calculation of segmentation metrics without accumulating raw data
            seg_tp_total += TP.sum().item()
            seg_tn_total += TN.sum().item()
            seg_fp_total += FP.sum().item()
            seg_fn_total += FN.sum().item()
            
            # Calculate current batch AUC (if there are positive and negative samples)
            try:
                batch_seg_preds = predict_mask_prob.cpu().numpy().flatten()
                batch_seg_gts = masks.cpu().numpy().flatten()
                if len(np.unique(batch_seg_gts)) > 1:
                    batch_auc = roc_auc_score(batch_seg_gts, batch_seg_preds)
                    seg_auc_scores.append(batch_auc)
            except:
                pass  # Skip failed AUC calculations
            
            # Save prediction results
            if args.save_predictions:
                for i in range(images.shape[0]):
                    filename = filenames[i] if isinstance(filenames, list) else f"img_{total_images + i}"
                    tamper_pred = tamper_prob[i, 0].cpu().item() > 0.5
                    save_prediction_image(
                        images[i], predict_mask_prob[i], masks[i],
                        tamper_pred, filename, prediction_dir
                    )
            
            total_images += images.shape[0]
    
    print("Inference completed, calculating metrics...")
    
    # Same final processing as test_one_epoch
    final_metrics = {}
    
    # Classification metrics calculation
    if len(all_tamper_preds) > 0:
        all_tamper_preds_cat = torch.cat(all_tamper_preds, dim=0).squeeze()
        all_tamper_gts_cat = torch.cat(all_tamper_gts, dim=0).squeeze()
        
        # Classification accuracy
        clf_preds_binary = (all_tamper_preds_cat > 0.5).float()
        clf_accuracy = (clf_preds_binary == all_tamper_gts_cat).float().mean().item()
        
        # Classification F1 and AUC
        clf_preds_np = clf_preds_binary.numpy()
        clf_gts_np = all_tamper_gts_cat.numpy()
        clf_probs_np = all_tamper_preds_cat.numpy()
        
        final_metrics['clf_acc'] = clf_accuracy
        final_metrics['clf_f1'] = f1_score(clf_gts_np, clf_preds_np, zero_division=0)
        if len(np.unique(clf_gts_np)) > 1:
            final_metrics['clf_auc'] = roc_auc_score(clf_gts_np, clf_probs_np)
        else:
            final_metrics['clf_auc'] = 0.0
    
    # Segmentation metrics calculation - using accumulated confusion matrix statistics
    if seg_tp_total + seg_tn_total + seg_fp_total + seg_fn_total > 0:
        # Segmentation IoU = TP / (TP + FP + FN)
        final_metrics['seg_iou'] = seg_tp_total / (seg_tp_total + seg_fp_total + seg_fn_total + 1e-8)
        
        # Segmentation AUC - using average of each batch's AUC
        if len(seg_auc_scores) > 0:
            final_metrics['seg_auc'] = np.mean(seg_auc_scores)
        else:
            final_metrics['seg_auc'] = 0.0
    else:
        final_metrics['seg_iou'] = 0.0
        final_metrics['seg_auc'] = 0.0
    
    # Segmentation F1 - using metric_logger results (custom evaluation method)
    final_metrics['seg_f1'] = metric_logger.meters['average_f1'].global_avg
    
    # Basic statistics
    final_metrics['total_images'] = total_images
    
    # Print results
    print("\n" + "="*50)
    print("Inference Test Results")
    print("="*50)
    print(f"Total images: {final_metrics['total_images']}")
    print("\nSegmentation Task Metrics:")
    print(f"  F1 Score: {final_metrics['seg_f1']:.4f}")
    print(f"  AUC: {final_metrics['seg_auc']:.4f}")
    print(f"  IoU: {final_metrics['seg_iou']:.4f}")
    print("\nClassification Task Metrics:")
    if 'clf_acc' in final_metrics:
        print(f"  Accuracy: {final_metrics['clf_acc']:.4f}")
        print(f"  F1 Score: {final_metrics['clf_f1']:.4f}")
        print(f"  AUC: {final_metrics['clf_auc']:.4f}")
    else:
        print("  Classification metrics not calculated (may not have valid classification data)")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'inference_results.json')
    with open(results_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    if args.save_predictions:
        print(f"Prediction images saved to: {prediction_dir}")
    
    return final_metrics


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    print("Inference Test Parameters:")
    print(f"Model path: {args.model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"ConvNeXt variant: {args.convnext_variant}")
    print(f"Device: {args.device}")
    print("-" * 50)
    
    # Run inference
    results = run_inference(args)
    print("\nInference test completed!")


if __name__ == '__main__':
    main()
