# 🚀 IGQK SaaS Platform – Deployment Guide (MVP/Demo, nicht production-ready)

**Version:** 3.0
**Status:** MVP/Demo (not production-ready)
**Last Updated:** 2026-02-04 (Status-Korrektur: 2026-10-26)

> ## ⚠️ Production-Blocker (Stand Code, 2026-10-26)
>
> Die Anleitung zeigt, **wie man den Docker-Stack startet** – sie macht das
> System nicht production-ready. Fünf Befunde im Code widersprechen dem:
>
> | # | Befund | Ort |
> |---|---|---|
> | 1 | JWT-`SECRET_KEY` hardcodiert, liest **keine** Umgebungsvariable | `backend/api/auth.py:19` |
> | 2 | CORS `allow_origins=["*"]` (mit Credentials) | `backend/main.py:26` |
> | 3 | Trainings-Jobs nur im Prozess-Dict (kein Restart/Multi-Worker) | `backend/api/training.py:72` |
> | 4 | Deployment-API liefert **501** für alle Cloud-Ziele außer `igqk_api` | `backend/api/deployment.py:58` |
> | 5 | SQLite statt Server-Datenbank | `backend/database.py:3-6` |
>
> **Konsequenz:** Die `SECRET_KEY` / `JWT_SECRET_KEY`-Einstellungen weiter unten
> sind derzeit wirkungslos. Punkte 1–2 müssen zuerst im Code behoben und danach
> erneut auditiert werden.

---

## 📋 TABLE OF CONTENTS

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Deployment](#deployment)
6. [Monitoring](#monitoring)
7. [Scaling](#scaling)
8. [Troubleshooting](#troubleshooting)
9. [Security](#security)
10. [Backup & Recovery](#backup--recovery)

---

## ⚡ QUICK START

### Windows (Recommended)

```bash
# 1. Clone repository
git clone <your-repo>
cd igqk_saas

# 2. Copy environment template
copy .env.example .env

# 3. Edit .env with your settings
notepad .env

# 4. Deploy!
deploy.bat
```

### Linux/Mac

```bash
# 1. Clone repository
git clone <your-repo>
cd igqk_saas

# 2. Copy environment template
cp .env.example .env

# 3. Edit .env with your settings
nano .env

# 4. Make scripts executable
chmod +x deploy.sh

# 5. Deploy!
./deploy.sh
```

**System will be available at:**
- Frontend: http://localhost:7860
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/api/docs

---

## 🔧 PREREQUISITES

### Required

✅ **Docker Desktop** (v20.10+)
- Download: https://www.docker.com/products/docker-desktop
- Minimum: 8GB RAM, 50GB Disk Space
- Recommended: 16GB RAM, 100GB Disk Space

✅ **Docker Compose** (v2.0+)
- Included with Docker Desktop
- Or install separately: https://docs.docker.com/compose/install/

### Optional (for Cloud Deployment)

- **AWS CLI** - For AWS deployment
- **Google Cloud SDK** - For GCP deployment
- **Azure CLI** - For Azure deployment
- **kubectl** - For Kubernetes deployment

---

## 📦 INSTALLATION

### 1. System Requirements

```
Minimum:
- CPU: 4 cores
- RAM: 8 GB
- Disk: 50 GB free
- Network: 10 Mbps

Recommended:
- CPU: 8 cores
- RAM: 16 GB
- Disk: 100 GB SSD
- Network: 100 Mbps
```

### 2. Install Docker

**Windows:**
1. Download Docker Desktop from https://docker.com
2. Run installer
3. Restart computer
4. Verify: `docker --version`

**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and back in
docker --version
```

**Mac:**
1. Download Docker Desktop from https://docker.com
2. Drag to Applications
3. Launch Docker
4. Verify: `docker --version`

### 3. Clone Repository

```bash
git clone <your-repository-url>
cd igqk_saas
```

---

## ⚙️ CONFIGURATION

### 1. Environment Variables

Copy the example environment file:

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

### 2. Edit Configuration

Open `.env` and configure:

#### **Essential Settings**

```env
# Environment
ENV=production
DEBUG=false

# Backend
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
BACKEND_WORKERS=4

# Frontend
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false

# Database
DATABASE_URL=sqlite:///./igqk_saas.db

# Security
SECRET_KEY=<generate-random-key>          # wird derzeit NICHT vom Backend gelesen
JWT_SECRET_KEY=<generate-random-key>      # wird derzeit NICHT vom Backend gelesen
```

> ⚠️ `backend/api/auth.py:19` verwendet einen hardcodierten `SECRET_KEY` und
> ignoriert diese Variablen. Sie vor dem Deployment zu setzen ist trotzdem
> sinnvoll (Vorbereitung), ändert aber aktuell nichts am Verhalten.

#### **Generate Secret Keys**

```bash
# Python
python -c "import secrets; print(secrets.token_hex(32))"

# Or OpenSSL
openssl rand -hex 32
```

#### **HuggingFace Token** (Optional but Recommended)

```env
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx
```

Get token from: https://huggingface.co/settings/tokens

### 3. Storage Configuration

Default directories:
- `./compressed_models` - Compressed models
- `./models_cache` - HuggingFace cache
- `./datasets_cache` - Dataset cache
- `./igqk_saas.db` - SQLite database

For production, consider:
- External SSD/NVMe for models
- S3/GCS for cloud storage
- Separate database server

---

## 🚀 DEPLOYMENT

### Local Deployment (Development/Testing)

#### Windows

```bash
# Option 1: Quick Deploy
deploy.bat

# Option 2: Management Interface
manage.bat
# Then select: 1. Start System
```

#### Linux/Mac

```bash
# Make executable
chmod +x deploy.sh

# Deploy
./deploy.sh
```

### Docker-Compose Deployment (Demo/Testbetrieb)

#### 1. Build Images

```bash
docker-compose build --no-cache
```

#### 2. Start Services

```bash
docker-compose up -d
```

#### 3. Verify Health

```bash
# Backend health
curl http://localhost:8000/api/health

# Frontend health
curl http://localhost:7860

# View logs
docker-compose logs -f
```

#### 4. Check Status

```bash
docker-compose ps
```

Expected output:
```
NAME              STATUS          PORTS
igqk-backend      Up (healthy)    0.0.0.0:8000->8000/tcp
igqk-frontend     Up (healthy)    0.0.0.0:7860->7860/tcp
```

---

## 📊 MONITORING

### Container Logs

```bash
# All logs
docker-compose logs -f

# Backend only
docker-compose logs -f backend

# Frontend only
docker-compose logs -f frontend

# Last 100 lines
docker-compose logs --tail=100
```

### Health Checks

```bash
# Backend API health
curl http://localhost:8000/api/health

# Response should be:
# {"status":"healthy","version":"3.0.0"}

# Frontend health
curl -I http://localhost:7860
# Should return HTTP 200
```

### Resource Usage

```bash
# Container stats
docker stats

# Disk usage
docker system df

# Network
docker network ls
```

### Application Metrics

Access built-in metrics:
- Backend: http://localhost:8000/api/stats
- Prometheus (if enabled): http://localhost:9090

---

## 📈 SCALING

### Horizontal Scaling (Multiple Instances)

Edit `docker-compose.yml`:

```yaml
backend:
  deploy:
    replicas: 4  # Run 4 backend instances

frontend:
  deploy:
    replicas: 2  # Run 2 frontend instances
```

### Vertical Scaling (More Resources)

```yaml
backend:
  deploy:
    resources:
      limits:
        cpus: '4.0'
        memory: 8G
      reservations:
        cpus: '2.0'
        memory: 4G
```

### Load Balancing

Use Nginx reverse proxy:

```bash
# Enable nginx profile
docker-compose --profile production up -d
```

> ⚠️ `docker-compose.yml` bindet `./nginx/nginx.conf` und `./nginx/ssl` ein,
> aber ein `nginx/`-Verzeichnis liegt **nicht** im Repo. Der Profil-Start
> schlägt daher fehl bzw. mountet leere Verzeichnisse. Nginx-Config + Zertifikate
> müssen erst erstellt werden.

---

## 🐛 TROUBLESHOOTING

### Common Issues

#### 1. Port Already in Use

```bash
# Check what's using port 8000
netstat -ano | findstr :8000

# Kill process (Windows)
taskkill /F /PID <PID>

# Kill process (Linux/Mac)
kill -9 <PID>
```

#### 2. Docker Build Fails

```bash
# Clean Docker cache
docker system prune -af

# Rebuild
docker-compose build --no-cache
```

#### 3. Container Won't Start

```bash
# View logs
docker-compose logs backend

# Check container
docker inspect igqk-backend

# Restart
docker-compose restart backend
```

#### 4. Database Locked

```bash
# Stop all containers
docker-compose down

# Remove volumes
docker-compose down -v

# Restart
docker-compose up -d
```

#### 5. Out of Memory

```bash
# Increase Docker memory limit in Docker Desktop settings
# Recommended: 8GB minimum, 16GB for production

# Or reduce workers in .env:
BACKEND_WORKERS=2
```

### Debug Mode

Enable debug logging:

```env
# .env
DEBUG=true
LOG_LEVEL=DEBUG
```

Restart:
```bash
docker-compose restart
```

---

## 🔒 SECURITY

### 1. Change Default Secrets

**Wichtig:** Diese Variablen haben derzeit **keine Wirkung** – der Key steht
hardcodiert in `backend/api/auth.py:19`. Erst nach dem Code-Fix (Key aus Env,
Fail-Closed bei Platzhalter) und Key-Rotation ist dieser Schritt wirksam.

```env
SECRET_KEY=<your-random-secret>
JWT_SECRET_KEY=<your-random-jwt-secret>
```

### 2. Enable HTTPS

Use Nginx with SSL:

```bash
# Generate self-signed certificate (testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem

# Or use Let's Encrypt (production)
# See: https://letsencrypt.org/
```

### 3. Firewall Rules

```bash
# Allow only necessary ports
# - 80 (HTTP)
# - 443 (HTTPS)
# - Block 8000, 7860 directly
```

### 4. API Rate Limiting

Configure in `.env`:

```env
MAX_CONCURRENT_JOBS=5
JOB_TIMEOUT_MINUTES=60
```

### 5. Database Security

```bash
# Set file permissions (Linux/Mac)
chmod 600 igqk_saas.db

# Backup regularly
# See Backup section below
```

---

## 💾 BACKUP & RECOVERY

### Automated Backup

Create `backup.sh`:

```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./backups/$DATE"

mkdir -p $BACKUP_DIR

# Backup database
cp igqk_saas.db $BACKUP_DIR/

# Backup compressed models
tar -czf $BACKUP_DIR/models.tar.gz compressed_models/

# Backup config
cp .env $BACKUP_DIR/

echo "Backup created: $BACKUP_DIR"
```

### Manual Backup

```bash
# Stop containers
docker-compose down

# Create backup
mkdir backups/$(date +%Y%m%d)
cp igqk_saas.db backups/$(date +%Y%m%d)/
cp -r compressed_models backups/$(date +%Y%m%d)/

# Restart
docker-compose up -d
```

### Restore from Backup

```bash
# Stop containers
docker-compose down

# Restore database
cp backups/20260204/igqk_saas.db ./

# Restore models
cp -r backups/20260204/compressed_models ./

# Restart
docker-compose up -d
```

---

## 🌐 CLOUD DEPLOYMENT

> **Abgrenzung:** Die App-interne Deployment-API unterstützt nur das Ziel
> `igqk_api`; AWS/GCP/Azure liefern **501** (`backend/api/deployment.py:58`).
> Die folgenden Schritte beschreiben das Starten des Docker-Stacks auf einer
> Cloud-VM – nicht das API-Feature.

### AWS EC2

```bash
# 1. Launch EC2 instance (t3.large or larger)
# 2. Install Docker
# 3. Clone repository
# 4. Configure .env
# 5. Deploy

# Security Group: Open ports 80, 443
```

### Google Cloud Platform

```bash
# 1. Create Compute Engine VM (n1-standard-4 or larger)
# 2. Install Docker
# 3. Clone repository
# 4. Configure .env
# 5. Deploy

# Firewall: Allow tcp:80,443
```

### Azure

```bash
# 1. Create VM (Standard_D4s_v3 or larger)
# 2. Install Docker
# 3. Clone repository
# 4. Configure .env
# 5. Deploy

# NSG: Inbound rules for 80, 443
```

### Kubernetes (Advanced)

Es liegen **keine** Kubernetes-Manifeste im Repo (`kubernetes/` existiert nicht,
Stand 2026-10-26). Kubernetes wäre erst nach eigener Erstellung der Manifeste
möglich.

---

## 📞 SUPPORT & MAINTENANCE

### Regular Maintenance

#### Daily
- Check logs for errors
- Monitor disk space
- Verify health endpoints

#### Weekly
- Review application metrics
- Check for security updates
- Test backups

#### Monthly
- Update Docker images
- Clean old model files
- Database optimization

### Update Application

```bash
# Pull latest code
git pull

# Rebuild images
docker-compose build --no-cache

# Restart with new images
docker-compose down
docker-compose up -d
```

### Performance Tuning

#### Backend Workers

```env
# Adjust based on CPU cores
# Formula: (2 × cores) + 1
BACKEND_WORKERS=9  # For 4-core CPU
```

#### Database Optimization

```bash
# SQLite VACUUM (clean up)
sqlite3 igqk_saas.db "VACUUM;"
```

#### Cache Management

```bash
# Clear HuggingFace cache
rm -rf models_cache/*

# Clear old compressed models
find compressed_models -mtime +30 -delete
```

---

## ✅ CHECKLIST VOR EINEM ECHTEN PRODUCTION-GO-LIVE

**Nicht abhakbar mit dem aktuellen Code:** Punkte 1 und 2 erfordern zuerst
Code-Änderungen (Env-Key, CORS). Der Rest setzt eine funktionierende
Betriebsumgebung voraus, die dieses Repo nicht mitbringt.

- [ ] Secret-Key aus Env gelesen + rotiert (**Code-Fix nötig**, `auth.py:19`)
- [ ] CORS auf konkrete Origins beschränkt (**Code-Fix nötig**, `main.py:26`)
- [ ] Job-Persistenz statt Prozess-Dict (`training.py:72`)
- [ ] Server-DB statt SQLite (`database.py:3-6`)
- [ ] Deployment-Ziele außer `igqk_api` implementiert (`deployment.py:58`)
- [ ] Nginx-Config + SSL-Zertifikate erstellt (`nginx/` fehlt)
- [ ] Firewall-Regeln gesetzt
- [ ] Backups konfiguriert und Wiederherstellung getestet
- [ ] Monitoring eingerichtet
- [ ] Lasttest durchgeführt
- [ ] Team geschult

---

## 📚 ADDITIONAL RESOURCES

- **API Documentation:** http://localhost:8000/api/docs
- **Docker Docs:** https://docs.docker.com
- **FastAPI Docs:** https://fastapi.tiangolo.com
- **Gradio Docs:** https://gradio.app/docs

---

## 🎉 DEMO-DEPLOYMENT LÄUFT (NICHT PRODUCTION-READY)

Der Docker-Stack läuft als Demo/MVP-Deployment. Ein Production-Status ist erst
nach den Blockpunkten am Dokumentanfang vertretbar.

**Access Points:**
- **Frontend UI:** http://localhost:7860
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/api/docs

**Management:**
- Start: `deploy.bat` or `./deploy.sh`
- Manage: `manage.bat`
- Logs: `docker-compose logs -f`
- Stop: `docker-compose down`

---

**For support, open an issue or contact the development team.**
