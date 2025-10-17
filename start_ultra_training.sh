#!/bin/bash

# Ultra-optimized training startup script for 2-week completion

echo "🚀 Starting Ultra-Optimized VMMRdb Training"
echo "Target: 50 epochs in 2 weeks on full dataset"
echo "=============================================="

# Apply system optimizations
echo "Applying system optimizations..."
python3 optimize_system.py

# Set ultra-optimized environment variables
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMBA_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTORCH_JIT=1
export PYTORCH_JIT_USE_NNC_NOT_NVFUSER=1

# Create necessary directories
mkdir -p data/output/logs
mkdir -p data/output/trials
mkdir -p lightning_logs

# Check system resources
echo "System Information:"
echo "CPU cores: $(sysctl -n hw.ncpu)"
echo "Memory: $(system_profiler SPHardwareDataType | grep 'Memory:' | awk '{print $2, $3}')"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
else
    echo "GPU: Not available (using CPU/MPS)"
fi

echo ""
echo "🎯 Ultra-Optimizations Applied:"
echo "- Batch size: 128 (GPU) / 32 (CPU)"
echo "- Workers: 16 (GPU) / 8 (CPU)"
echo "- Gradient accumulation: 2x"
echo "- Validation: 10% of data"
echo "- Checkpointing: Every 2 epochs"
echo "- Mixed precision: 16-bit"
echo "- System optimizations: Maximum"
echo ""

# Start ultra-optimized training
echo "Starting ultra-optimized training at $(date)"
python3 train_vmmrdb_ultra_optimized.py
