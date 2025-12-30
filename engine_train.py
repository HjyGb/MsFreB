# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
"""
Training and evaluation engine for the image manipulation detection model.

This module contains the functions for running a single training epoch (`train_one_epoch`)
and a single evaluation epoch (`test_one_epoch`). It handles metric logging,
gradient scaling, learning rate scheduling, and TensorBoard logging.

Based on the training engines from MAE, DeiT, and BEiT.
"""
import math
import sys
from typing import Iterable

import torch

import utils.misc as misc
import utils.lr_sched as lr_sched

from utils.datasets import denormalize
import utils.evaluation as evaluation

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    """
    Runs a single training epoch.

    Args:
        model (torch.nn.Module): The model to train.
        data_loader (Iterable): The data loader for the training set.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to run on (e.g., 'cuda').
        epoch (int): The current epoch number.
        loss_scaler: The gradient scaler for mixed-precision training.
        log_writer (SummaryWriter, optional): Logger for TensorBoard. Defaults to None.
        args: Command-line arguments.

    Returns:
        dict: A dictionary of averaged training statistics for the epoch.
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('focal_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('edge_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('dice_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('clf_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # Add a meter for consistency loss if it's being used.
    if args.consistency_loss_weight > 0:
        metric_logger.add_meter('cons_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, masks, edge_masks, shape) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # Adjust learning rate per iteration.
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        edge_masks = edge_masks.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            # The model returns a tuple of loss components and predictions.
            outputs = model(samples, masks, edge_masks, shape)
            total_loss = outputs[0]
            pred_mask = outputs[1]
            tamper_pred_prob = outputs[2]
            focal_loss_val = outputs[3]
            edge_loss_val = outputs[4]
            dice_loss_val = outputs[5]
            classification_loss_val = outputs[6]
            consistency_loss_val = outputs[7]

        loss_value = total_loss.item()
        focal_loss_item = focal_loss_val.item()
        edge_loss_item = edge_loss_val.item()
        dice_loss_item = dice_loss_val.item()
        classification_loss_item = classification_loss_val.item()
        consistency_loss_item = consistency_loss_val.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        total_loss = total_loss / accum_iter 
        loss_scaler(total_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0, clip_grad = args.clip_grad)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(total_loss=loss_value)
        metric_logger.update(focal_loss=focal_loss_item)
        metric_logger.update(edge_loss=edge_loss_item)
        metric_logger.update(dice_loss=dice_loss_item)
        metric_logger.update(clf_loss=classification_loss_item)
        if args.consistency_loss_weight > 0:
            metric_logger.update(cons_loss=consistency_loss_item)
        
        # Reduce losses for logging across all processes.
        focal_loss_reduce = misc.all_reduce_mean(focal_loss_item)
        edge_loss_reduce = misc.all_reduce_mean(edge_loss_item)
        dice_loss_reduce = misc.all_reduce_mean(dice_loss_item)
        classification_loss_reduce = misc.all_reduce_mean(classification_loss_item)
        consistency_loss_reduce = misc.all_reduce_mean(consistency_loss_item) if args.consistency_loss_weight > 0 else 0.0
        total_loss_reduced_for_logging = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % 50 == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss/focal_loss', focal_loss_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss/edge_loss', edge_loss_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss/dice_loss', dice_loss_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss/clf_loss', classification_loss_reduce, epoch_1000x)
            if args.consistency_loss_weight > 0: # Log if active
                log_writer.add_scalar('train_loss/cons_loss', consistency_loss_reduce, epoch_1000x)
            
            log_writer.add_scalar('train_loss/total_loss', total_loss_reduced_for_logging, epoch_1000x)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Log images to TensorBoard at the end of the epoch.
    if log_writer is not None and misc.is_main_process():
        log_writer.add_images('train/image',  denormalize(samples), epoch)
        log_writer.add_images('train/predict', pred_mask, epoch) # Log raw probability map.
        log_writer.add_images('train/predict_t', (pred_mask > 0.5) * 1.0, epoch) # Log thresholded prediction.
        log_writer.add_images('train/masks', masks, epoch)
        log_writer.add_images('train/edge_mask', edge_masks, epoch)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items() if k != 'total_loss'}


def test_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    device: torch.device, 
                    epoch: int, 
                    log_writer=None,
                    args=None):
    """
    Runs a single evaluation epoch.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (Iterable): The data loader for the test set.
        device (torch.device): The device to run on.
        epoch (int): The current epoch number.
        log_writer (SummaryWriter, optional): Logger for TensorBoard. Defaults to None.
        args: Command-line arguments.

    Returns:
        dict: A dictionary of averaged evaluation statistics for the epoch.
    """
    with torch.no_grad(): 
        model.zero_grad() 
        model.eval()      
        metric_logger = misc.MetricLogger(delimiter="  ")
        if args.consistency_loss_weight > 0:
            metric_logger.add_meter('cons_loss_test', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        print_freq = 20
        header = 'Test: [{}]'.format(epoch)
        all_tamper_preds = []
        all_tamper_gts = []

        for data_iter_step, (images, masks, edge_mask, shape) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            edge_mask = edge_mask.to(device, non_blocking=True) 

            # In evaluation, we are primarily interested in the predictions.
            # The model call still computes losses, which can be logged for diagnostics.
            outputs = model(images, masks, edge_mask, shape)
            predict_mask_prob = outputs[1]
            tamper_prob = outputs[2]
            clf_loss_val = outputs[6]
            consistency_loss_eval = outputs[7]
            
            predict_mask_prob = predict_mask_prob.detach()
            tamper_prob = tamper_prob.detach()

            # Determine image-level ground truth for classification.
            is_tampered_gt_batch = (masks.sum(dim=[1, 2, 3]) > 1e-5).float().unsqueeze(1)
            all_tamper_preds.append(tamper_prob.cpu())
            all_tamper_gts.append(is_tampered_gt_batch.cpu())

            #---- Segmentation Evaluation (F1 Score) ----
            region_mask = evaluation.genertate_region_mask(masks, shape) 
            TP, TN, FP, FN = evaluation.cal_confusion_matrix(predict_mask_prob, masks, region_mask)
            local_f1 = evaluation.cal_F1(TP, TN, FP, FN) 
            
            for i in local_f1: 
                metric_logger.update(average_f1=i)
            metric_logger.update(clf_loss_test=clf_loss_val.item())
            if args.consistency_loss_weight > 0:
                 metric_logger.update(cons_loss_test=consistency_loss_eval.item())


        # After the loop, calculate classification metrics from all batches.
        if len(all_tamper_preds) > 0:
            all_tamper_preds_cat = torch.cat(all_tamper_preds, dim=0).squeeze()
            all_tamper_gts_cat = torch.cat(all_tamper_gts, dim=0).squeeze()
            
            # Calculate binary accuracy for classification.
            clf_preds_binary = (all_tamper_preds_cat > 0.5).float()
            clf_accuracy = (clf_preds_binary == all_tamper_gts_cat).float().mean().item()
            metric_logger.synchronize_between_processes() # Synchronize before accessing global_avg
            metric_logger.add_meter('clf_acc_test', misc.SmoothedValue(window_size=1, fmt='{value:.4f}')) # Add meter before update
            metric_logger.update(clf_acc_test=clf_accuracy) # Update with the calculated accuracy
            if log_writer is not None and misc.is_main_process():
                 log_writer.add_scalar('clf/test_accuracy', clf_accuracy, epoch)
        # Synchronize metrics across all processes.
        metric_logger.synchronize_between_processes()    
        
        # Log results after synchronization.
        if log_writer is not None and misc.is_main_process(): 
            log_writer.add_scalar('F1/test_average', metric_logger.meters['average_f1'].global_avg, epoch)
            if 'clf_loss_test' in metric_logger.meters: 
                 log_writer.add_scalar('loss/test_clf_loss', metric_logger.meters['clf_loss_test'].global_avg, epoch)
            if args.consistency_loss_weight > 0 and 'cons_loss_test' in metric_logger.meters:
                 log_writer.add_scalar('loss/test_cons_loss', metric_logger.meters['cons_loss_test'].global_avg, epoch)
            
            # Log images from the last processed batch.
            log_writer.add_images('test/image',  denormalize(images), epoch) 
            log_writer.add_images('test/predict_mask', (predict_mask_prob > 0.5)* 1.0, epoch) 
            log_writer.add_images('test/masks', masks, epoch)
            log_writer.add_images('test/edge_mask', edge_mask, epoch)
            # log_writer.add_scalar('test/tamper_prob_mean_last_batch', tamper_prob.mean(), epoch)

        print("Averaged stats:", metric_logger)
        
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
