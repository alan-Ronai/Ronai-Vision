#!/bin/bash
set -e

INSTANCE_IP=$1
KEY_FILE=$2

if [ -z "$INSTANCE_IP" ] || [ -z "$KEY_FILE" ]; then
    echo "Usage: ./deploy_to_ec2.sh <instance-ip> <key-file>"
    echo ""
    echo "Example:"
    echo "  ./deploy_to_ec2.sh 54.123.45.67 ~/.ssh/my-key.pem"
    exit 1
fi

echo "========================================"
echo "Ronai-Vision AWS EC2 Deployment"
echo "========================================"
echo "Target: ubuntu@$INSTANCE_IP"
echo "Key: $KEY_FILE"
echo ""

# Upload application
echo "[1/5] Uploading application files..."
rsync -avz -e "ssh -i $KEY_FILE -o StrictHostKeyChecking=no" \
    --exclude 'venv' \
    --exclude '*.pyc' \
    --exclude '__pycache__' \
    --exclude 'audio_storage' \
    --exclude 'output' \
    --exclude '.git' \
    --exclude '.gitignore' \
    --exclude '*.pt' \
    --exclude '*.engine' \
    --exclude 'models' \
    ./ ubuntu@$INSTANCE_IP:/home/ubuntu/Ronai-Vision/

echo ""
echo "[2/5] Installing Python dependencies..."
ssh -i $KEY_FILE ubuntu@$INSTANCE_IP << 'EOF'
cd /home/ubuntu/Ronai-Vision
source venv/bin/activate
pip install -r requirements.txt
EOF

echo ""
echo "[3/5] Creating required directories..."
ssh -i $KEY_FILE ubuntu@$INSTANCE_IP << 'EOF'
mkdir -p /home/ubuntu/Ronai-Vision/audio_storage/recordings
mkdir -p /home/ubuntu/Ronai-Vision/audio_storage/sessions
mkdir -p /home/ubuntu/Ronai-Vision/output
mkdir -p /home/ubuntu/Ronai-Vision/logs
EOF

echo ""
echo "[4/5] Restarting service..."
ssh -i $KEY_FILE ubuntu@$INSTANCE_IP << 'EOF'
sudo systemctl restart ronai-vision
sleep 2
sudo systemctl status ronai-vision --no-pager
EOF

echo ""
echo "[5/5] Verifying deployment..."
ssh -i $KEY_FILE ubuntu@$INSTANCE_IP << 'EOF'
curl -s http://localhost:8000/api/health || echo "Health check failed"
EOF

echo ""
echo "========================================"
echo "Deployment complete!"
echo "========================================"
echo ""
echo "Access points:"
echo "  - API: http://$INSTANCE_IP:8000"
echo "  - Health: http://$INSTANCE_IP:8000/api/health"
echo "  - Audio Status: http://$INSTANCE_IP:8000/api/audio/status"
echo "  - RTSP: rtsp://$INSTANCE_IP:8554/audio"
echo ""
echo "Useful commands:"
echo "  - View logs: ssh -i $KEY_FILE ubuntu@$INSTANCE_IP 'sudo journalctl -u ronai-vision -f'"
echo "  - Check status: ssh -i $KEY_FILE ubuntu@$INSTANCE_IP 'sudo systemctl status ronai-vision'"
echo "  - Restart: ssh -i $KEY_FILE ubuntu@$INSTANCE_IP 'sudo systemctl restart ronai-vision'"
echo ""
