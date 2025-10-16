# Google Cloud Platform Deployment Guide

This guide will help you deploy and train the EfficientNet model on Google Cloud Platform.

## Prerequisites

1. **Google Cloud Account**: Set up a GCP account and enable billing
2. **Google Cloud SDK**: Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
3. **Docker**: Install Docker for building container images
4. **Project Setup**: Create a GCP project and enable required APIs

## Setup Steps

### 1. Initialize Google Cloud

```bash
# Login to Google Cloud
gcloud auth login

# Set your project ID (replace with your actual project ID)
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable container.googleapis.com
```

### 2. Create a Cloud Storage Bucket

```bash
# Create a bucket for storing data and model outputs
gsutil mb gs://$PROJECT_ID-efficientnet-data

# Upload your data (if you have it locally)
# gsutil -m cp -r data/input/stanford gs://$PROJECT_ID-efficientnet-data/
```

### 3. Build and Push Docker Image

```bash
# Build the Docker image
docker build -t gcr.io/$PROJECT_ID/efficientnet-stanford-cars .

# Push to Google Container Registry
docker push gcr.io/$PROJECT_ID/efficientnet-stanford-cars
```

### 4. Training Options

#### Option A: Using AI Platform Training (Recommended)

Create a training job configuration file:

```yaml
# training-job.yaml
trainingInput:
  scaleTier: CUSTOM
  masterType: n1-standard-4
  masterConfig:
    imageUri: gcr.io/PROJECT_ID/efficientnet-stanford-cars
    acceleratorConfig:
      type: NVIDIA_TESLA_T4
      count: 1
  region: us-central1
  args: ["python", "train_gcp.py"]
```

Submit the training job:

```bash
gcloud ai-platform jobs submit training efficientnet-training-$(date +%Y%m%d-%H%M%S) \
  --config training-job.yaml \
  --region us-central1
```

#### Option B: Using Compute Engine

Create a VM instance:

```bash
# Create VM with GPU
gcloud compute instances create efficientnet-training \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=tf-latest-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --restart-on-failure \
  --scopes=https://www.googleapis.com/auth/cloud-platform

# SSH into the instance
gcloud compute ssh efficientnet-training --zone=us-central1-a
```

On the VM, run:

```bash
# Clone your repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Install dependencies
pip install -r requirements-gcp.txt

# Run training
python train_gcp.py
```

#### Option C: Using Vertex AI (Latest)

```bash
# Create a custom training job
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name="efficientnet-stanford-cars" \
  --config=vertex-ai-config.yaml
```

Create `vertex-ai-config.yaml`:

```yaml
displayName: "EfficientNet Stanford Cars Training"
jobSpec:
  workerPoolSpecs:
  - machineSpec:
      machineType: n1-standard-4
      acceleratorType: NVIDIA_TESLA_T4
      acceleratorCount: 1
    replicaCount: 1
    containerSpec:
      imageUri: gcr.io/PROJECT_ID/efficientnet-stanford-cars
      command: ["python"]
      args: ["train_gcp.py"]
```

## Monitoring Training

### View Logs

```bash
# For AI Platform jobs
gcloud ai-platform jobs describe JOB_ID --region=us-central1

# For Compute Engine
gcloud compute instances get-serial-port-output efficientnet-training --zone=us-central1-a
```

### TensorBoard

If using TensorBoard logging, you can view logs by:

1. Downloading logs from the training job
2. Running TensorBoard locally: `tensorboard --logdir=lightning_logs`

## Cost Optimization Tips

1. **Use Preemptible Instances**: Add `--preemptible` flag for 60-80% cost savings
2. **Choose Right Machine Type**: Start with smaller instances and scale up if needed
3. **Set Up Budget Alerts**: Monitor spending in GCP Console
4. **Use Spot Instances**: For Vertex AI, use spot instances for additional savings

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size in `train_gcp.py`
2. **Permission Errors**: Ensure proper IAM roles are assigned
3. **Data Download Issues**: Check network connectivity and URLs
4. **GPU Not Detected**: Verify accelerator configuration

### Useful Commands

```bash
# Check GPU availability
nvidia-smi

# Monitor resource usage
htop

# Check disk space
df -h

# View recent logs
tail -f /var/log/syslog
```

## Cleanup

After training, clean up resources to avoid charges:

```bash
# Delete VM instance
gcloud compute instances delete efficientnet-training --zone=us-central1-a

# Delete training job (if using AI Platform)
gcloud ai-platform jobs cancel JOB_ID --region=us-central1

# Delete Docker images
gcloud container images delete gcr.io/$PROJECT_ID/efficientnet-stanford-cars
```

## Estimated Costs

- **n1-standard-4 + T4 GPU**: ~$0.35/hour
- **Training time**: ~8-12 hours for 150 epochs
- **Total estimated cost**: $3-5 per training run

*Note: Costs may vary based on region, machine type, and training duration.*
