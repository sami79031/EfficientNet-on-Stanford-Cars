# VMMRdb Training Optimization Summary

## 🎯 Goal
Train EfficientNet-B0 on VMMRdb dataset (285K images, 9,169 classes) for 50 epochs within 3-7 days.

## 🚀 Key Optimizations

### 1. **Batch Size Optimization**
- **GPU**: Batch size 64 (optimal for memory and speed)
- **CPU**: Batch size 16 (memory-safe)
- **Auto-calculation**: Based on dataset size and target batches per epoch

### 2. **Data Loading Optimization**
- **Workers**: 8 for GPU, 4 for CPU
- **Persistent workers**: Keep workers alive between epochs
- **Prefetch factor**: 2 batches ahead
- **Pin memory**: Enabled for GPU training
- **Drop last**: Consistent batch sizes

### 3. **Training Optimizations**
- **Mixed precision**: 16-bit for GPU, bfloat16 for CPU
- **Gradient clipping**: 1.0 for stability
- **Checkpointing**: Every 5 epochs + best model
- **Early stopping**: 20 epochs patience
- **Logging**: Every 50 steps (not every step)

### 4. **Memory Optimizations**
- **cuDNN benchmark**: Enabled for consistent input sizes
- **Non-deterministic**: Allowed for speed
- **Gradient accumulation**: Disabled (single GPU)

## 📊 Expected Performance

### **With GPU (Recommended)**
- **Time per epoch**: ~2-3 hours
- **Total time**: 4-6 days
- **Memory usage**: ~8-12 GB VRAM
- **Throughput**: ~0.1 seconds per batch

### **With CPU (Fallback)**
- **Time per epoch**: ~20-30 hours
- **Total time**: 40-60 days
- **Memory usage**: ~4-8 GB RAM
- **Throughput**: ~0.5 seconds per batch

## 🛠️ Usage

### **Start Training**
```bash
# Make script executable (already done)
chmod +x run_optimized_training.sh

# Start optimized training
./run_optimized_training.sh
```

### **Monitor Training**
```bash
# Check training status
python monitor_training.py

# Monitor continuously
watch -n 30 python monitor_training.py

# View live logs
tail -f data/output/logs/training_*.log

# TensorBoard (if available)
tensorboard --logdir lightning_logs
```

### **Check Progress**
```bash
# Check if training is running
ps aux | grep python

# Check GPU usage
nvidia-smi

# Check training results
ls -la data/output/trials/
ls -la lightning_logs/
```

## 📁 Files Created

1. **`train_vmmrdb_optimized.py`** - Main optimized training script
2. **`training/train_optimized.py`** - Optimized training function
3. **`run_optimized_training.sh`** - Background execution script
4. **`monitor_training.py`** - Training monitoring tool

## 🔧 Configuration

### **Automatic Settings**
- Batch size calculated based on dataset size
- Workers optimized for CPU/GPU
- Precision automatically selected
- Checkpointing frequency optimized

### **Manual Overrides**
You can modify these in `train_vmmrdb_optimized.py`:
- `batch_size`: Override automatic calculation
- `epochs`: Change from 50 to any number
- `num_workers`: Adjust data loading workers
- `precision`: Change mixed precision settings

## 📈 Monitoring

The training will create:
- **Logs**: `data/output/logs/training_YYYYMMDD_HHMMSS.log`
- **Checkpoints**: `data/output/trials/[trial_id]/`
- **TensorBoard**: `lightning_logs/[trial_id]/`
- **PID file**: `training.pid`

## ⚠️ Important Notes

1. **GPU Recommended**: CPU training will take 40-60 days
2. **Memory**: Ensure sufficient RAM/VRAM
3. **Storage**: ~50GB free space for checkpoints and logs
4. **Network**: Stable connection for long training
5. **Power**: Ensure uninterrupted power supply

## 🎉 Expected Results

After 50 epochs, you should have:
- **Model accuracy**: 85-95% on validation set
- **Training time**: 3-7 days (with GPU)
- **Model size**: ~15MB (EfficientNet-B0)
- **Checkpoints**: Multiple saved models for deployment

## 🚨 Troubleshooting

### **Training Stops**
```bash
# Check if process is running
ps aux | grep python

# Restart if needed
./run_optimized_training.sh
```

### **Out of Memory**
- Reduce batch size in `train_vmmrdb_optimized.py`
- Reduce number of workers
- Use gradient accumulation

### **Slow Training**
- Check GPU usage: `nvidia-smi`
- Verify mixed precision is enabled
- Check data loading bottlenecks

---

**Ready to train! 🚀**
