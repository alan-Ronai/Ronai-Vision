# AWS EC2 Deployment Guide

## Overview

This guide covers deploying Ronai-Vision (including the RTP/RTSP audio server) to AWS EC2.

## Architecture Options

### Option 1: Single EC2 Instance (Recommended for POC)
- All services on one instance
- Simpler setup, lower cost
- Good for testing and small deployments

### Option 2: Multi-Instance Setup (Production)
- Separate instances for video processing and audio ingestion
- Better scalability and isolation
- Higher cost

**This guide covers Option 1.**

---

## Prerequisites

1. **AWS Account** with permissions to:
   - Create EC2 instances
   - Create Security Groups
   - Create Elastic IPs (optional)

2. **Local Tools**:
   - AWS CLI (`aws configure`)
   - SSH key pair for EC2 access

3. **Domain/DNS** (optional):
   - For production, point a domain to your EC2 IP

---

## Step 1: Launch EC2 Instance

### Instance Specifications

**For CPU-based processing (Development/POC):**
- **Instance Type**: `t3.xlarge` or `t3.2xlarge`
- **vCPUs**: 4-8
- **Memory**: 16-32 GB
- **Storage**: 100 GB SSD (gp3)

**For GPU-based processing (Production):**
- **Instance Type**: `g4dn.xlarge` or `g5.xlarge`
- **GPU**: NVIDIA T4 or A10G
- **vCPUs**: 4-8
- **Memory**: 16-32 GB
- **Storage**: 200 GB SSD (gp3)

### AMI (Operating System)

Choose one:
- **Ubuntu 22.04 LTS** (Recommended)
- **Amazon Linux 2023**
- **Deep Learning AMI** (GPU instances, has CUDA pre-installed)

### Launch via AWS Console

1. Go to EC2 Console → Launch Instance
2. **Name**: `ronai-vision-server`
3. **AMI**: Ubuntu 22.04 LTS
4. **Instance Type**: `t3.xlarge` (or `g4dn.xlarge` for GPU)
5. **Key Pair**: Select or create SSH key
6. **Network Settings**:
   - Create new security group or use existing
   - See Security Group configuration below
7. **Storage**: 100 GB gp3 SSD
8. **Advanced Details** → User Data (optional, see automation script below)

### Launch via AWS CLI

```bash
# Set variables
KEY_NAME="your-key-pair-name"
SECURITY_GROUP="sg-xxxxxxxxx"  # Your security group ID
SUBNET="subnet-xxxxxxxxx"       # Your subnet ID

# Launch instance
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type t3.xlarge \
  --key-name $KEY_NAME \
  --security-group-ids $SECURITY_GROUP \
  --subnet-id $SUBNET \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ronai-vision-server}]'
```

---

## Step 2: Configure Security Group

### Required Inbound Rules

| Type | Protocol | Port Range | Source | Description |
|------|----------|------------|--------|-------------|
| SSH | TCP | 22 | Your IP | SSH access |
| HTTP | TCP | 8000 | 0.0.0.0/0 | FastAPI server |
| Custom TCP | TCP | 8554 | 0.0.0.0/0 | RTSP server |
| Custom UDP | UDP | 5004-5005 | 0.0.0.0/0 | RTP audio |
| Custom TCP | TCP | 554 | 0.0.0.0/0 | RTSP (standard port, optional) |

### Create Security Group via CLI

```bash
# Create security group
aws ec2 create-security-group \
  --group-name ronai-vision-sg \
  --description "Security group for Ronai Vision server"

# Get the security group ID
SG_ID=$(aws ec2 describe-security-groups \
  --group-names ronai-vision-sg \
  --query 'SecurityGroups[0].GroupId' \
  --output text)

# Add inbound rules
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 22 --cidr YOUR_IP/32
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 8000 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 8554 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol udp --port 5004-5005 --cidr 0.0.0.0/0
```

---

## Step 3: Connect to EC2 Instance

```bash
# Get instance public IP
INSTANCE_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=ronai-vision-server" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

# SSH into instance
ssh -i ~/.ssh/your-key.pem ubuntu@$INSTANCE_IP
```

---

## Step 4: Install Dependencies on EC2

### Update System

```bash
sudo apt update
sudo apt upgrade -y
```

### Install Python and Essential Tools

```bash
# Install Python 3.11+
sudo apt install -y python3.11 python3.11-venv python3-pip

# Install system dependencies
sudo apt install -y \
  git \
  ffmpeg \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  build-essential
```

### Install CUDA (GPU Instances Only)

```bash
# For g4dn/g5 instances
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvidia-smi
```

---

## Step 5: Deploy Application

### Clone Repository

```bash
cd /home/ubuntu
git clone https://github.com/your-username/Ronai-Vision.git
cd Ronai-Vision
```

**Or upload via SCP:**

```bash
# From your local machine
scp -i ~/.ssh/your-key.pem -r /Users/alankantor/Downloads/Ronai/Ronai-Vision ubuntu@$INSTANCE_IP:/home/ubuntu/
```

### Create Virtual Environment

```bash
cd /home/ubuntu/Ronai-Vision
python3.11 -m venv venv
source venv/bin/activate
```

### Install Python Dependencies

```bash
# Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or for GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### Download Model Weights

```bash
mkdir -p models
cd models

# Download YOLO model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O yolo12n.pt

# Download SAM2 model (small)
# Follow SAM2 installation instructions or download checkpoint

# Download OSNet ReID model
# Follow torchreid instructions

cd ..
```

---

## Step 6: Configure Production Environment

### Create Production Config

```bash
# Copy and edit production env file
cp config/prod.env config/prod.env.local
nano config/prod.env.local
```

**Update settings:**

```bash
DEVICE=cuda  # or cpu
CAMERA_CONFIG=config/camera_settings.json
SAVE_FRAMES=false
PARALLEL=true
ALLOWED_CLASSES="person"

# RTP/RTSP Audio Server
AUDIO_SERVER_ENABLED=true
AUDIO_CONFIG=config/audio_settings.json
AUDIO_STORAGE_PATH=/mnt/audio_storage/recordings
```

### Create Audio Storage Directory

```bash
sudo mkdir -p /mnt/audio_storage/recordings
sudo chown ubuntu:ubuntu /mnt/audio_storage
```

### Update Audio Config for Production

```bash
nano config/audio_settings.json
```

Update port to standard RTSP port (554) if desired:

```json
{
  "rtsp": {
    "host": "0.0.0.0",
    "port": 554,
    "session_timeout": 60,
    "keepalive_interval": 30
  }
}
```

**Note**: Port 554 requires root. Use systemd service with capabilities (see below).

---

## Step 7: Create Systemd Service

### Create Service File

```bash
sudo nano /etc/systemd/system/ronai-vision.service
```

**Service configuration:**

```ini
[Unit]
Description=Ronai Vision Server
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/Ronai-Vision
Environment="PATH=/home/ubuntu/Ronai-Vision/venv/bin"
EnvironmentFile=/home/ubuntu/Ronai-Vision/config/prod.env.local
ExecStart=/home/ubuntu/Ronai-Vision/venv/bin/uvicorn api.server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

# Allow binding to privileged ports (554)
AmbientCapabilities=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
```

### Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable ronai-vision

# Start service
sudo systemctl start ronai-vision

# Check status
sudo systemctl status ronai-vision

# View logs
sudo journalctl -u ronai-vision -f
```

---

## Step 8: Setup Nginx Reverse Proxy (Optional)

### Install Nginx

```bash
sudo apt install -y nginx
```

### Configure Nginx

```bash
sudo nano /etc/nginx/sites-available/ronai-vision
```

**Nginx configuration:**

```nginx
server {
    listen 80;
    server_name your-domain.com;  # Or use IP

    # API endpoints
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # WebSocket
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }

    # Health check
    location /health {
        proxy_pass http://localhost:8000/api/health;
    }
}
```

**Enable site:**

```bash
sudo ln -s /etc/nginx/sites-available/ronai-vision /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## Step 9: Setup SSL/TLS (Production)

### Install Certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
```

### Obtain Certificate

```bash
sudo certbot --nginx -d your-domain.com
```

### Auto-renewal

```bash
sudo certbot renew --dry-run
```

---

## Step 10: Monitoring and Logging

### Install Monitoring Tools

```bash
# Install htop
sudo apt install -y htop

# Install nvidia-smi for GPU monitoring (if applicable)
watch -n 1 nvidia-smi
```

### Application Logs

```bash
# View application logs
sudo journalctl -u ronai-vision -f

# View nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# View audio recordings
ls -lh /mnt/audio_storage/recordings/
```

### Setup Log Rotation

```bash
sudo nano /etc/logrotate.d/ronai-vision
```

```
/var/log/ronai-vision/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 ubuntu ubuntu
    sharedscripts
}
```

---

## Step 11: Test Deployment

### Test API

```bash
# From local machine
curl http://$INSTANCE_IP:8000/api/health
curl http://$INSTANCE_IP:8000/api/audio/status

# Start audio server
curl -X POST http://$INSTANCE_IP:8000/api/audio/start
```

### Test RTSP/RTP Audio

```bash
# From local machine, send test audio
python scripts/test_audio_client.py --host $INSTANCE_IP --port 8554 --duration 10

# Check recording via API
curl http://$INSTANCE_IP:8000/api/audio/recordings
```

### Test with VLC

```
vlc rtsp://$INSTANCE_IP:8554/audio
```

---

## Automated Deployment Script

Create `scripts/deploy_to_ec2.sh`:

```bash
#!/bin/bash
set -e

INSTANCE_IP=$1
KEY_FILE=$2

if [ -z "$INSTANCE_IP" ] || [ -z "$KEY_FILE" ]; then
    echo "Usage: ./deploy_to_ec2.sh <instance-ip> <key-file>"
    exit 1
fi

echo "Deploying to $INSTANCE_IP..."

# Upload application
rsync -avz -e "ssh -i $KEY_FILE" \
    --exclude 'venv' \
    --exclude '*.pyc' \
    --exclude '__pycache__' \
    --exclude 'audio_storage' \
    --exclude '.git' \
    ./ ubuntu@$INSTANCE_IP:/home/ubuntu/Ronai-Vision/

# Run setup commands
ssh -i $KEY_FILE ubuntu@$INSTANCE_IP << 'EOF'
cd /home/ubuntu/Ronai-Vision
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart ronai-vision
EOF

echo "Deployment complete!"
echo "Access: http://$INSTANCE_IP:8000"
```

**Usage:**

```bash
chmod +x scripts/deploy_to_ec2.sh
./scripts/deploy_to_ec2.sh 1.2.3.4 ~/.ssh/your-key.pem
```

---

## Cost Optimization

### Instance Sizing

| Use Case | Instance Type | Cost/Month* |
|----------|---------------|-------------|
| Development/POC | t3.xlarge | ~$120 |
| Production CPU | t3.2xlarge | ~$240 |
| Production GPU | g4dn.xlarge | ~$350 |

*Approximate costs for us-east-1 region

### Savings Strategies

1. **Spot Instances**: 70-90% savings for non-critical workloads
2. **Reserved Instances**: 30-60% savings with 1-3 year commitment
3. **Auto-scaling**: Scale down during off-hours
4. **S3 for Storage**: Move old recordings to S3 (cheaper than EBS)

---

## Troubleshooting

### Application Won't Start

```bash
# Check service status
sudo systemctl status ronai-vision

# Check logs
sudo journalctl -u ronai-vision --no-pager

# Test manually
cd /home/ubuntu/Ronai-Vision
source venv/bin/activate
python -c "import services.audio; print('OK')"
```

### Port Binding Issues

```bash
# Check if port is in use
sudo netstat -tlnp | grep 8554

# Check capabilities
getcap /home/ubuntu/Ronai-Vision/venv/bin/python3

# Grant capabilities
sudo setcap 'cap_net_bind_service=+ep' /home/ubuntu/Ronai-Vision/venv/bin/python3
```

### Connection Refused

1. Check security group rules
2. Verify service is running: `sudo systemctl status ronai-vision`
3. Check firewall: `sudo ufw status`
4. Test locally: `curl http://localhost:8000/api/health`

---

## Security Best Practices

1. **Use IAM Roles** instead of access keys
2. **Restrict Security Groups** to known IPs when possible
3. **Enable AWS GuardDuty** for threat detection
4. **Regular Updates**: `sudo apt update && sudo apt upgrade`
5. **Rotate SSH Keys** regularly
6. **Enable AWS CloudWatch** for logging
7. **Use AWS Secrets Manager** for sensitive config

---

## Backup Strategy

### EBS Snapshots

```bash
# Create snapshot
aws ec2 create-snapshot \
    --volume-id vol-xxxxxxxxx \
    --description "Ronai Vision backup $(date +%Y-%m-%d)"
```

### Application Data

```bash
# Backup audio recordings to S3
aws s3 sync /mnt/audio_storage/recordings/ s3://your-bucket/audio-backups/
```

---

## Next Steps

✅ Instance launched and configured
✅ Application deployed and running
✅ Audio server tested and verified
✅ Monitoring and logging setup
✅ SSL/TLS configured (optional)

**Production Checklist:**
- [ ] Setup CloudWatch alarms
- [ ] Configure backup strategy
- [ ] Load testing
- [ ] Disaster recovery plan
- [ ] Documentation for team

---

**Ready for Production Deployment!** 🚀
