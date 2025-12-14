#!/bin/bash
# Check AWS networking configuration for RTP

echo "=========================================="
echo "AWS Network Diagnostics for RTP"
echo "=========================================="
echo ""

# Check if running on EC2
if [ -f /sys/hypervisor/uuid ] && [ `head -c 3 /sys/hypervisor/uuid` == "ec2" ]; then
    echo "✓ Running on AWS EC2"
    echo ""
else
    echo "⚠️  Not running on EC2 (or can't detect)"
    echo ""
fi

# Get instance metadata (if on EC2)
echo "Instance Information:"
echo "--------------------"
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null)
if [ ! -z "$TOKEN" ]; then
    PUBLIC_IP=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null)
    PRIVATE_IP=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/local-ipv4 2>/dev/null)
    INSTANCE_ID=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null)
    AZ=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/placement/availability-zone 2>/dev/null)
    
    echo "Instance ID: $INSTANCE_ID"
    echo "Public IP: $PUBLIC_IP"
    echo "Private IP: $PRIVATE_IP"
    echo "Availability Zone: $AZ"
else
    echo "Could not fetch instance metadata"
    echo "Public IP: $(curl -s ifconfig.me 2>/dev/null || echo "unknown")"
    echo "Private IP: $(hostname -I | awk '{print $1}')"
fi
echo ""

# Check listening ports
echo "Listening Ports:"
echo "--------------------"
netstat -uln | grep -E '5004|8554' || echo "No RTP/RTSP ports listening"
echo ""

# Check firewall (iptables)
echo "Firewall Rules (iptables):"
echo "--------------------"
if command -v iptables &> /dev/null; then
    sudo iptables -L INPUT -n | grep -E '5004|8554|ACCEPT.*udp' | head -5 || echo "No specific RTP rules found"
else
    echo "iptables not available"
fi
echo ""

# Check if firewalld is running
if systemctl is-active --quiet firewalld 2>/dev/null; then
    echo "⚠️  firewalld is running - may block traffic"
    sudo firewall-cmd --list-all 2>/dev/null | grep -E 'ports|services'
    echo ""
fi

# Network interfaces
echo "Network Interfaces:"
echo "--------------------"
ip addr show | grep -E '^[0-9]:|inet ' | head -10 || ifconfig | grep -E '^[a-z]|inet ' | head -10
echo ""

# Routing table
echo "Default Route:"
echo "--------------------"
ip route show default || route -n | grep '^0.0.0.0'
echo ""

# Test UDP socket binding
echo "Testing UDP Socket Binding:"
echo "--------------------"
python3 << 'EOF'
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', 5004))
    print("✓ Can bind to UDP 0.0.0.0:5004")
    s.close()
except Exception as e:
    print(f"❌ Cannot bind to UDP 5004: {e}")
EOF
echo ""

echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Verify Security Group allows UDP 5004 inbound"
echo "2. Check Network ACL (subnet level) allows UDP 5004"
echo "3. Ask sender to confirm destination IP: ${PUBLIC_IP:-[your-ip]}"
echo "4. Run: sudo tcpdump -i any -n udp port 5004"
echo "5. Run: python scripts/debug_rtp_receiver.py --port 5004"
echo "=========================================="
