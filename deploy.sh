#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------
# EC2 Bootstrap Script for Automated Trading System
#
# Run on a fresh Amazon Linux 2023 EC2 instance:
#   chmod +x deploy.sh && ./deploy.sh
#
# Prerequisites:
#   - EC2 t3.small (2 vCPU, 2 GB RAM) in ap-south-1 with SG allowing SSH (22) and HTTP (8000)
#   - Add CloudWatch agent for CPU/memory alerts (recommended)
#   - RDS PostgreSQL instance created and accessible from this EC2
#   - .env file prepared with real credentials (see .env.example)
# -----------------------------------------------------------------------

APP_DIR="$HOME/trading-app/automated_trading_app"
SWAP_SIZE_MB=1024

echo "=== 1/5  System packages ==="
sudo yum update -y
sudo yum install -y docker git

echo "=== 2/5  Docker engine ==="
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker "$USER"

# Docker Compose plugin
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

echo "=== 3/5  Swap file (${SWAP_SIZE_MB} MB) ==="
if [ ! -f /swapfile ]; then
  sudo dd if=/dev/zero of=/swapfile bs=1M count=$SWAP_SIZE_MB
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
  echo "Swap enabled."
else
  echo "Swap already exists, skipping."
fi

echo "=== 4/5  Alembic migrations ==="
if [ ! -f "$APP_DIR/.env" ]; then
  echo "ERROR: $APP_DIR/.env not found."
  echo "Copy .env.example to .env and fill in real credentials before running this script."
  exit 1
fi

source "$APP_DIR/.env"

if [ -z "${DATABASE_URL:-}" ]; then
  echo "ERROR: DATABASE_URL not set in .env"
  exit 1
fi

sudo yum install -y python3.12 python3.12-pip
pip3.12 install --user sqlalchemy psycopg2-binary alembic pyyaml
cd "$APP_DIR"
export DATABASE_URL
alembic upgrade head
echo "Migrations applied."

echo "=== 5/5  Start containers ==="
# newgrp docker would start a subshell; use sudo for the first run instead
sudo docker compose up -d --build
echo ""
echo "Containers started. Verify with:"
echo "  sudo docker compose logs -f"
echo ""
echo "Health check:"
echo "  curl http://localhost:8000/health"
