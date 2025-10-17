#!/bin/bash

# Optimized training script for VMMRdb dataset
# Designed to complete 50 epochs in 3-7 days

echo "=========================================="
echo "Starting Optimized VMMRdb Training"
echo "Target: 50 epochs in 3-7 days"
echo "=========================================="

# Create necessary directories
mkdir -p data/output/logs
mkdir -p data/output/trials
mkdir -p lightning_logs

# Set environment variables for optimization
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

# Check system resources
echo "System Information:"
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
else
    echo "GPU: Not available"
fi

# Start training with proper logging
echo "Starting training at $(date)"
echo "Log file: data/output/logs/training_$(date +%Y%m%d_%H%M%S).log"

# Run training in background with nohup
nohup python train_vmmrdb_optimized.py > "data/output/logs/training_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

# Get the process ID
TRAINING_PID=$!

echo "Training started with PID: $TRAINING_PID"
echo "PID saved to: training.pid"

# Save PID for monitoring
echo $TRAINING_PID > training.pid

echo ""
echo "=========================================="
echo "Training Commands:"
echo "=========================================="
echo "Monitor progress:"
echo "  tail -f data/output/logs/training_*.log"
echo ""
echo "Check if training is running:"
echo "  ps aux | grep $TRAINING_PID"
echo ""
echo "Monitor GPU usage (if available):"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Check training metrics:"
echo "  tensorboard --logdir lightning_logs"
echo ""
echo "Stop training:"
echo "  kill $TRAINING_PID"
echo ""
echo "Check training results:"
echo "  ls -la data/output/trials/"
echo "  ls -la lightning_logs/"
echo "=========================================="

# Show initial log output
echo "Initial training output:"
sleep 5
tail -n 20 "data/output/logs/training_$(date +%Y%m%d_%H%M%S).log" 2>/dev/null || echo "Log file not yet created"
