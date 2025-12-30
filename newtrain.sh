#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=1,3 && python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 /DATA/shared1/MsFreB/MsFreB/main_train.py \
--world_size 2 \
--batch_size 1 \
--convnext_variant base \
--convnext_pretrain_path /DATA/shared1/MsFreB/MsFreB/pretrained-weights/convnextv2_base_1k_224_ema.pt \
--data_path /DATA/shared1/MsFreB/MsFreB/Dataset/CASIA2.0/ \
--epochs 200 \
--lr 1e-5 \
--min_lr 5e-7 \
--weight_decay 0.05 \
--edge_lambda 20 \
--focal_alpha 0.1 \
--test_data_path /DATA/shared1/MsFreB/MsFreB/Dataset/CASIA1.0/ \
--warmup_epochs 4 \
--output_dir /DATA/shared1/MsFreB/MsFreB/output/ \
--log_dir /DATA/shared1/MsFreB/MsFreB/output/ \
--accum_iter 1 \
--seed 42 \
--test_period 4 \
--num_workers 0 \
1> /DATA/shared1/MsFreB/MsFreB/train_log.log 2> /DATA/shared1/MsFreB/MsFreB/train_error.log

if [ $? -eq 0 ]; then
    echo "训练已正常结束！"
else
    echo "? 训练异常退出！请查看错误日志：/DATA/shared1/MsFreB/MsFreB/train_error.log"
fi
