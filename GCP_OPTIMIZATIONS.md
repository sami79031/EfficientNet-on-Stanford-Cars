# GCP Optimizations for EfficientNet Training

This document outlines the key optimizations made to run EfficientNet training on Google Cloud Platform instead of Mac.

## Key Changes Made

### 1. **Training Script Optimizations** (`training/train_gcp.py`)

**Mac → GCP Changes:**
- **Batch Size**: 16 → 64 (4x increase for GPU)
- **Mixed Precision**: Added `precision=16` for faster training
- **Gradient Clipping**: Added `gradient_clip_val=1.0` for stability
- **Sanity Checks**: Re-enabled `num_sanity_val_steps=2` (was 0 on Mac)
- **Logging**: Added `log_every_n_steps=50` for better cloud monitoring
- **Weight Decay**: Added `weight_decay=0.01` to optimizer settings
- **Scheduler**: More aggressive learning rate reduction

### 2. **Data Loading Optimizations** (`datasets/stanford/stanford_cars_data_module_gcp.py`)

**Mac → GCP Changes:**
- **Workers**: `num_workers=0` → `num_workers=8` (parallel data loading)
- **Persistent Workers**: Added `persistent_workers=True` (keep workers alive)
- **Prefetch Factor**: Added `prefetch_factor=2` (prefetch batches)
- **Drop Last**: Added `drop_last=True` (consistent batch sizes)
- **Augmentation**: Re-enabled `RandomErasing` (was commented out on Mac)
- **More Aggressive Augmentation**: Increased rotation, translation, and color jitter

### 3. **Environment Optimizations** (`train_gcp_optimized.py`)

**New Features:**
- **Dynamic Batch Size**: Automatically determines optimal batch size based on GPU memory
- **CUDA Optimizations**: 
  - `torch.backends.cudnn.benchmark = True`
  - `torch.backends.cudnn.deterministic = False`
- **Memory Management**: `torch.cuda.empty_cache()` to prevent OOM
- **GPU Detection**: Better GPU availability checking

### 4. **Performance Improvements**

| Setting | Mac | GCP | Improvement |
|---------|-----|-----|-------------|
| Batch Size | 16 | 64-128 | 4-8x |
| Data Workers | 0 | 8 | 8x parallel loading |
| Mixed Precision | No | Yes | ~2x speed |
| Memory Optimizations | Conservative | Aggressive | Better utilization |

## Usage

### For VM Training:
```bash
python train_gcp_optimized.py
```

### For AI Platform Training:
```bash
python training/train_gcp.py
```

## Expected Performance Gains

- **Training Speed**: 3-5x faster on GPU vs CPU
- **Memory Efficiency**: Better GPU memory utilization
- **Data Loading**: 8x faster with parallel workers
- **Convergence**: Better with larger batch sizes and mixed precision

## GPU Memory Requirements

| GPU Type | Memory | Recommended Batch Size |
|----------|--------|----------------------|
| T4 | 16 GB | 64 |
| V100 | 32 GB | 128 |
| A100 | 40-80 GB | 128-256 |
| CPU | N/A | 16 |

## Cost Optimization

- **Preemptible Instances**: 60-80% cost savings
- **Spot Instances**: Additional 20-30% savings
- **Auto-scaling**: Only pay for actual training time
- **Storage**: Use regional persistent disks for better performance

## Monitoring

The GCP-optimized version includes:
- TensorBoard logging for real-time monitoring
- GPU utilization tracking
- Memory usage monitoring
- Training progress visualization

## Troubleshooting

### Common Issues:
1. **OOM Errors**: Reduce batch size in `get_optimal_batch_size()`
2. **Slow Data Loading**: Check `num_workers` setting
3. **GPU Not Detected**: Restart VM or check driver installation
4. **Memory Issues**: Enable gradient checkpointing or reduce model size
