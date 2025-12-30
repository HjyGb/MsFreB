#!/bin/bash

# Model testing script - benchmark performance then evaluate accuracy
# Usage: ./test.sh

echo "=========================================="
echo "Starting Model Testing Pipeline"
echo "=========================================="

echo ""
echo "Step 1: Performance Benchmark"
echo "Testing inference speed and memory usage with random weights..."
echo "------------------------------------------"

python -u benchmark.py \
  --device cuda \
  --input_size 1024 \
  --batch_sizes 1 2 4 8 \
  --num_iterations 50 \
  --warmup_steps 10 \
  2> benchmark_error.log 1> benchmark_log.log

if [ $? -eq 0 ]; then
    echo "✓ Benchmark completed successfully"
else
    echo "✗ Benchmark failed, check benchmark_error.log"
    exit 1
fi

echo ""
echo "Step 2: Accuracy Evaluation"
echo "Testing model accuracy with trained weights..."
echo "------------------------------------------"

python -u inference.py \
  --model_path "./output_convnext/checkpoint-190.pth" \
  --data_path "./Dataset/CASIA1.0" \
  --output_dir "./inference_results" \
  --batch_size 2 \
  --num_workers 4 \
  --device cuda \
  --save_predictions \
  --seed 42 \
  2> inference_error.log 1> inference_log.log

if [ $? -eq 0 ]; then
    echo "✓ Inference evaluation completed successfully"
else
    echo "✗ Inference evaluation failed, check inference_error.log"
    exit 1
fi

echo ""
echo "=========================================="
echo "Testing Pipeline Completed Successfully!"
echo "=========================================="
echo "Results:"
echo "- Benchmark logs: benchmark_log.log"
echo "- Inference logs: inference_log.log"
echo "- Detailed results: ./inference_results/"
echo "=========================================="
