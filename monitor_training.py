#!/usr/bin/env python3
"""
Training monitoring script for long-running VMMRdb training.
"""

import os
import time
import subprocess
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json

def get_training_status():
    """Get current training status."""
    status = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'running': False,
        'pid': None,
        'epoch': None,
        'loss': None,
        'accuracy': None,
        'gpu_usage': None,
        'memory_usage': None
    }
    
    # Check if training is running
    try:
        with open('training.pid', 'r') as f:
            pid = int(f.read().strip())
        
        # Check if process is still running
        result = subprocess.run(['ps', '-p', str(pid)], capture_output=True, text=True)
        if result.returncode == 0:
            status['running'] = True
            status['pid'] = pid
    except:
        pass
    
    # Get latest log file
    log_files = list(Path('data/output/logs').glob('training_*.log'))
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        
        # Parse latest log for metrics
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                
            # Look for latest epoch info
            for line in reversed(lines[-100:]):  # Check last 100 lines
                if 'Epoch' in line and 'train_loss' in line:
                    # Extract epoch number
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'Epoch':
                            status['epoch'] = parts[i+1].split(':')[0]
                            break
                    
                    # Extract loss and accuracy
                    if 'train_loss_step=' in line:
                        loss_start = line.find('train_loss_step=') + len('train_loss_step=')
                        loss_end = line.find(',', loss_start)
                        if loss_end == -1:
                            loss_end = line.find(' ', loss_start)
                        status['loss'] = line[loss_start:loss_end]
                    
                    if 'train_acc_step=' in line:
                        acc_start = line.find('train_acc_step=') + len('train_acc_step=')
                        acc_end = line.find(',', acc_start)
                        if acc_end == -1:
                            acc_end = line.find(' ', acc_start)
                        status['accuracy'] = line[acc_start:acc_end]
                    break
        except:
            pass
    
    # Get GPU usage if available
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(', ')
            if len(gpu_info) >= 3:
                status['gpu_usage'] = f"{gpu_info[0]}%"
                status['memory_usage'] = f"{gpu_info[1]}/{gpu_info[2]} MB"
    except:
        pass
    
    return status

def estimate_completion_time(current_epoch, total_epochs=50):
    """Estimate completion time based on current progress."""
    if current_epoch is None or current_epoch == 0:
        return "Unknown"
    
    try:
        current_epoch = int(current_epoch)
        progress = current_epoch / total_epochs
        
        # Estimate time per epoch (rough estimate)
        time_per_epoch_hours = 2.0  # Adjust based on your system
        
        remaining_epochs = total_epochs - current_epoch
        remaining_hours = remaining_epochs * time_per_epoch_hours
        remaining_days = remaining_hours / 24
        
        return f"{remaining_days:.1f} days ({remaining_hours:.1f} hours)"
    except:
        return "Unknown"

def main():
    """Main monitoring function."""
    print("=" * 60)
    print("VMMRdb Training Monitor")
    print("=" * 60)
    
    status = get_training_status()
    
    print(f"Timestamp: {status['timestamp']}")
    print(f"Training Running: {'Yes' if status['running'] else 'No'}")
    
    if status['pid']:
        print(f"Process ID: {status['pid']}")
    
    if status['epoch']:
        print(f"Current Epoch: {status['epoch']}/50")
        print(f"Progress: {float(status['epoch'])/50*100:.1f}%")
        print(f"Estimated Time Remaining: {estimate_completion_time(status['epoch'])}")
    
    if status['loss']:
        print(f"Current Loss: {status['loss']}")
    
    if status['accuracy']:
        print(f"Current Accuracy: {status['accuracy']}")
    
    if status['gpu_usage']:
        print(f"GPU Usage: {status['gpu_usage']}")
        print(f"GPU Memory: {status['memory_usage']}")
    
    print("\n" + "=" * 60)
    print("Recent Log Output:")
    print("=" * 60)
    
    # Show recent log output
    log_files = list(Path('data/output/logs').glob('training_*.log'))
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
            
            # Show last 10 lines
            for line in lines[-10:]:
                print(line.rstrip())
        except:
            print("Could not read log file")
    else:
        print("No log files found")
    
    print("\n" + "=" * 60)
    print("Commands:")
    print("=" * 60)
    print("Monitor continuously: watch -n 30 python monitor_training.py")
    print("View full log: tail -f data/output/logs/training_*.log")
    print("Check GPU: nvidia-smi")
    print("TensorBoard: tensorboard --logdir lightning_logs")

if __name__ == '__main__':
    main()
