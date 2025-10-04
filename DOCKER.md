# Docker Deployment Guide

Complete guide for deploying the Vietnamese Medical Chatbot System using Docker.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available
- 10GB disk space

## Quick Start

### 1. Clone and Configure

```bash
git clone https://github.com/ititiu20201/MedicalChatbot.git
cd MedicalChatbot

# Copy environment template
cp .env.example .env

# Edit .env with your Gemini API key
nano .env  # or use your preferred editor
```

### 2. Download Model Files

Place your PhoBERT model in the correct location:

```bash
# Create models directory
mkdir -p app/models

# Download or copy your model file
# cp /path/to/phobert_medchat_model.pt app/models/
```

### 3. Start Services

```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Access the Application

- **Patient Interface**: http://localhost:5500
- **Admin Dashboard**: http://localhost:5500/admin.html
- **API Documentation**: http://localhost:8003/docs
- **Health Check**: http://localhost:8003/health

## Docker Services

### Backend API (FastAPI)
- **Port**: 8003
- **Container**: medical_chatbot_backend
- **Purpose**: Main API server with PhoBERT and Gemini integration

### Frontend (Nginx)
- **Port**: 5500
- **Container**: medical_chatbot_frontend
- **Purpose**: Static file server for patient and admin interfaces

### Database (MySQL)
- **Port**: 3306
- **Container**: medical_chatbot_db
- **Purpose**: Patient records and medical data storage

## Environment Variables

Edit `.env` file to configure:

```env
# Database
DB_ROOT_PASSWORD=your_secure_password
DB_NAME=medical_chatbot
DB_USER=medchat
DB_PASSWORD=your_db_password

# Gemini AI
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-flash

# Database URL
DATABASE_URL=mysql+pymysql://medchat:password@mysql:3306/medical_chatbot
```

## Common Commands

### Start Services
```bash
docker-compose up -d
```

### Stop Services
```bash
docker-compose down
```

### Restart Services
```bash
docker-compose restart
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f mysql
```

### Rebuild After Code Changes
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Database Management
```bash
# Access MySQL shell
docker-compose exec mysql mysql -u root -p

# Backup database
docker-compose exec mysql mysqldump -u root -p medical_chatbot > backup.sql

# Restore database
docker-compose exec -T mysql mysql -u root -p medical_chatbot < backup.sql
```

## Volume Management

### Persistent Data
- `mysql_data`: Database storage
- `model_data`: PhoBERT model files

### List Volumes
```bash
docker volume ls
```

### Remove Volumes (⚠️ Deletes all data)
```bash
docker-compose down -v
```

## Production Deployment

### 1. Security Configuration

Update `docker-compose.yml` for production:

```yaml
# Use environment files
env_file:
  - .env.production

# Set resource limits
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

### 2. SSL/HTTPS Setup

Add SSL certificates to nginx:

```nginx
server {
    listen 443 ssl http2;
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    # ... rest of config
}
```

### 3. Update Frontend API Configuration

Edit `frontend/config.js`:

```javascript
const CONFIG = {
  API_BASE: 'https://your-domain.com/api',
  // ... other settings
};
```

## Troubleshooting

### Backend won't start
```bash
# Check logs
docker-compose logs backend

# Verify environment variables
docker-compose exec backend env | grep GEMINI

# Restart service
docker-compose restart backend
```

### Database connection failed
```bash
# Check MySQL is running
docker-compose ps mysql

# Test connection
docker-compose exec backend python -c "from app.db import engine; print(engine)"
```

### Model not found
```bash
# Verify model file exists
docker-compose exec backend ls -la app/models/

# Copy model into container
docker cp phobert_medchat_model.pt medical_chatbot_backend:/app/app/models/
```

### Port already in use
```bash
# Find process using port
lsof -i :8003  # or :5500, :3306

# Change port in docker-compose.yml
ports:
  - "8004:8003"  # Use different host port
```

## Health Checks

All services include health checks:

```bash
# Check backend health
curl http://localhost:8003/health

# Check frontend
curl http://localhost:5500

# Check MySQL
docker-compose exec mysql mysqladmin ping
```

## Performance Optimization

### 1. Build with BuildKit
```bash
DOCKER_BUILDKIT=1 docker-compose build
```

### 2. Multi-stage Build (Optional)
Edit `Dockerfile` for smaller image:

```dockerfile
FROM python:3.9-slim as builder
# Build dependencies
...

FROM python:3.9-slim
COPY --from=builder /app /app
...
```

### 3. Resource Allocation
```bash
# Increase Docker memory (Docker Desktop)
# Settings → Resources → Memory: 8GB
```

## Monitoring

### Container Stats
```bash
docker stats
```

### Resource Usage
```bash
docker-compose top
```

### Disk Usage
```bash
docker system df
```

## Cleanup

### Remove Stopped Containers
```bash
docker-compose down
```

### Remove All (including volumes)
```bash
docker-compose down -v
docker system prune -a
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/ititiu20201/MedicalChatbot/issues
- Check logs: `docker-compose logs -f`
- Verify configuration: Review `.env` and `docker-compose.yml`
