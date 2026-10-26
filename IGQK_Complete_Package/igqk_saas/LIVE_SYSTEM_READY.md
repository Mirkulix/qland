# ⚠️ IGQK SaaS – MVP/DEMO (NICHT PRODUCTION-READY)

**IGQK v3.0 SaaS Platform**
**Status:** 🟡 MVP/Demo – not production-ready
**Datum:** 2026-02-04 (Status-Korrektur: 2026-10-26)

> **Dieses Dokument beschreibt ein Demo-/MVP-System, kein production-ready System.**
> Der Code widerspricht einem Produktionsstatus an fünf Stellen:
>
> | # | Befund | Ort |
> |---|---|---|
> | 1 | JWT-`SECRET_KEY` ist im Code hardcodiert und wird nicht aus der Umgebung gelesen | `backend/api/auth.py:19` |
> | 2 | CORS erlaubt `*` (alle Origins, mit Credentials) | `backend/main.py:26` |
> | 3 | Trainings-Jobs liegen in einem prozesslokalen Dict (kein Restart, kein Multi-Worker) | `backend/api/training.py:72` |
> | 4 | Cloud-Deployment-Ziele liefern **501**; nur `igqk_api` (dieser Server) funktioniert | `backend/api/deployment.py:58` |
> | 5 | Datenbank ist SQLite (eine Datei, kein Server-Backend) | `backend/database.py:3-6` |
>
> Die `.env`-Variablen `SECRET_KEY`/`JWT_SECRET_KEY` aus diesem Dokument haben
> derzeit **keine Wirkung**. Erst nach Behebung von 1–2 (und erneuter Prüfung)
> ist ein Production-Status vertretbar.

---

## ✅ FERTIG FÜR DEMO/TEST – NICHT FÜR PRODUKTION

Das IGQK SaaS Platform ist ein lauffähiges Demo-System mit:
- Docker-Container + docker-compose-Orchestrierung
- Deployment- und Management-Scripts (lokal)
- Health Checks
- Backup-Anleitung (manuell)
- Konfigurationsvorlage (.env.example)

---

## 🚀 WIE SIE ES JETZT STARTEN

### **Methode 1: Lokales Demo-System (EMPFOHLEN)**

#### Windows

```bash
# 1. Kopieren Sie die Environment-Vorlage
copy .env.example .env

# 2. Editieren Sie .env (wichtig!)
notepad .env

# 3. Deployen Sie mit einem Klick!
deploy.bat
```

**Fertig!** System läuft in 5-10 Minuten!

#### Linux/Mac

```bash
# 1. Kopieren Sie die Environment-Vorlage
cp .env.example .env

# 2. Editieren Sie .env
nano .env

# 3. Deployen Sie!
chmod +x deploy.sh
./deploy.sh
```

**Fertig!** System läuft!

---

### **Methode 2: Docker auf einer Cloud-VM**

> **Abgrenzung:** Die App-interne Deployment-API (`POST /api/deployment/deploy`)
> unterstützt nur `igqk_api` und liefert für AWS/GCP/Azure **501**
> (`backend/api/deployment.py:58`). Was hier beschrieben wird, ist das Starten
> des Docker-Stacks auf einer VM per `deploy.sh` – nicht das API-Feature
> „Multi-Cloud-Deployment“. Ein Production-Betrieb bleibt wegen der fünf
> Blockpunkte am Dokumentanfang ausgeschlossen.

#### AWS EC2

```bash
# 1. Starten Sie EC2 Instance (t3.large+)
# 2. SSH verbinden
# 3. Install Docker:
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu

# 4. Clone Repository
git clone <your-repo>
cd igqk_saas

# 5. Configure
cp .env.example .env
nano .env

# 6. Deploy
./deploy.sh

# 7. Open ports in Security Group: 80, 443
```

**Ihre App läuft auf:** http://<ec2-public-ip>

#### Google Cloud Platform (GCP)

```bash
# 1. Create VM (n1-standard-4+)
# 2. SSH connect
# 3. Install Docker:
curl -fsSL https://get.docker.com | sh

# 4. Clone & Deploy wie oben
# 5. Firewall: Allow tcp:80,443
```

#### Azure

```bash
# 1. Create VM (Standard_D4s_v3+)
# 2. SSH connect
# 3. Install Docker
# 4. Clone & Deploy
# 5. NSG: Allow 80, 443
```

---

## 📦 WAS WURDE ERSTELLT

### **Docker-Container**

✅ **backend/Dockerfile**
- FastAPI Backend
- IGQK Core integriert
- Health Checks
- Multi-worker Support

✅ **Dockerfile** (Frontend)
- Gradio Web UI
- API-Client
- Fortschrittsanzeige

### **Orchestrierung**

✅ **docker-compose.yml**
- Backend Service
- Frontend Service
- Nginx (optional)
- Networking
- Volume Management
- Health Checks

### **Konfiguration**

✅ **.env.example**
- Alle Environment Variables
- Security Settings
- API Keys
- Limits & Quotas
- Cloud Integration (AWS/GCP/Azure)

### **Deployment-Scripts**

✅ **deploy.bat** (Windows)
- Automatisches Deployment
- Health Checks
- Browser-öffnung

✅ **deploy.sh** (Linux/Mac)
- Production Deployment
- Validation
- Status-Reporting

✅ **manage.bat** (Windows)
- Interaktives Management-Menu
- Start/Stop/Restart
- Logs anzeigen
- System Status
- Build & Clean

### **Dokumentation**

✅ **PRODUCTION_DEPLOYMENT.md**
- Komplette Deployment-Anleitung
- Troubleshooting
- Scaling
- Security
- Backup & Recovery

✅ **README.md**
- Quick Start
- Architektur
- API-Beispiele

---

## 🔧 SYSTEM-ARCHITEKTUR

```
┌─────────────────────────────────────────────┐
│            INTERNET / USERS                 │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│      Nginx Reverse Proxy (Optional)         │
│         Port 80/443 (HTTPS)                 │
└────────────────┬────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│  Frontend    │  │   Backend    │
│  (Gradio)    │  │  (FastAPI)   │
│  Port 7860   │  │  Port 8000   │
│              │  │              │
│ - Search UI  │  │ - REST API   │
│ - Progress   │  │ - Jobs       │
│ - Model Hub  │  │ - DB         │
└──────────────┘  └──────┬───────┘
                         │
                ┌────────┼────────┐
                │        │        │
                ▼        ▼        ▼
        ┌──────────┐ ┌────┐ ┌──────┐
        │   IGQK   │ │ DB │ │  HF  │
        │   Core   │ │SQL │ │ Hub  │
        └──────────┘ └────┘ └──────┘
                │        │        │
                ▼        ▼        ▼
        ┌───────────────────────────┐
        │       STORAGE             │
        │ - Models                  │
        │ - Cache                   │
        │ - Database                │
        └───────────────────────────┘
```

---

## 🎯 DEPLOYMENT-CHECKLISTE

### **Vor dem Deployment:**

- [x] Docker installiert
- [x] Docker Compose installiert
- [ ] .env-Datei konfiguriert
- [ ] SECRET_KEY generiert
- [ ] JWT_SECRET_KEY generiert
- [ ] Ports 8000 und 7860 frei
- [ ] Mindestens 8GB RAM verfügbar
- [ ] 50GB freier Speicherplatz

> ⚠️ **Hinweis:** `SECRET_KEY` / `JWT_SECRET_KEY` haben derzeit keine Wirkung,
> weil `backend/api/auth.py:19` einen hardcodierten Key verwendet. Diese Punkte
> sind erst nach dem Code-Fix sinnvoll abhakbar.

### **Deployment:**

- [ ] `deploy.bat` oder `./deploy.sh` ausgeführt
- [ ] Backend-Health-Check erfolgreich
- [ ] Frontend-Health-Check erfolgreich
- [ ] Browser geöffnet auf http://localhost:7860

### **Nach dem Deployment:**

- [ ] Test-Kompression durchgeführt
- [ ] Logs überprüft
- [ ] Model Hub getestet
- [ ] API-Docs zugänglich (http://localhost:8000/api/docs)
- [ ] Backup-Strategie eingerichtet

---

## 📊 FEATURES IM DEMO-SETUP

### **1. Automatisches Health Monitoring**

```yaml
healthcheck:
  test: curl -f http://localhost:8000/api/health
  interval: 30s
  timeout: 10s
  retries: 3
```

### **2. Auto-Restart bei Crashes**

```yaml
restart: unless-stopped
```

### **3. Resource Limits**

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
```

### **4. Volume Persistence**

```yaml
volumes:
  - ./backend/compressed_models:/app/backend/compressed_models
  - ./backend/igqk_saas.db:/app/backend/igqk_saas.db
```

### **5. Network Isolation**

```yaml
networks:
  igqk-network:
    driver: bridge
```

---

## 🔒 SECURITY – AKTUELLER STAND (TEILWEISE NUR DOKUMENTIERT)

### **1. Environment Variables**

- `.env.example` ist vorhanden
- ⚠️ **Aber:** `backend/api/auth.py:19` hardcodet
  `SECRET_KEY = "igqk_secret_key_change_in_production_2024"` und liest **keine**
  Umgebungsvariable. Jede Installation dieses Codes nutzt denselben Signing-Key.

### **2. CORS Protection**

Geplant:

```python
ALLOWED_ORIGINS=http://localhost:7860,https://your-domain.com
```

⚠️ **Aber:** `backend/main.py:26` setzt `allow_origins=["*"]` (mit
`allow_credentials=True`). Die Variable `ALLOWED_ORIGINS` wird nicht ausgewertet.

### **3. Rate Limiting**

```env
MAX_CONCURRENT_JOBS=5
JOB_TIMEOUT_MINUTES=60
```

### **4. JWT Authentication**

Implementiert (`backend/api/auth.py`), aber mit hardcodiertem Key:

```env
JWT_SECRET_KEY=your-secret   # wird derzeit nicht gelesen
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
```

**Fazit:** Für eine echte Produktion müssen 1, 2 und 4 zuerst im Code behoben
und danach erneut geprüft werden – nicht nur in `.env` gesetzt werden.

---

## 📈 SCALING-OPTIONEN (EINGESCHRÄNKT)

> ⚠️ Mehrere Backend-Replicas sind mit dem aktuellen Code **nicht** sicher:
> Trainings-Jobs liegen in einem prozesslokalen Dict
> (`backend/api/training.py:72`), d. h. Status/Ergebnisse eines Jobs sind nur in
> der Instanz sichtbar, die ihn gestartet hat. Horizontal Scaling erst nach
> Job-Persistenz (DB/Queue).

### **Horizontal Scaling**

```yaml
# docker-compose.yml
backend:
  deploy:
    replicas: 4  # 4 Backend-Instanzen
```

### **Vertical Scaling**

```yaml
backend:
  deploy:
    resources:
      limits:
        cpus: '8.0'
        memory: 16G
```

### **Load Balancing**

```bash
# Mit Nginx
docker-compose --profile production up -d
```

---

## 🛠️ MANAGEMENT

### **Start System**

```bash
# Windows
deploy.bat

# Linux/Mac
./deploy.sh
```

### **Stop System**

```bash
docker-compose down
```

### **View Logs**

```bash
# All services
docker-compose logs -f

# Backend only
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100
```

### **Restart Services**

```bash
docker-compose restart
```

### **Update System**

```bash
git pull
docker-compose build --no-cache
docker-compose down
docker-compose up -d
```

---

## 💾 BACKUP & RECOVERY

### **Automatisches Backup** (täglich empfohlen)

```bash
# Backup Script
DATE=$(date +%Y%m%d)
mkdir -p backups/$DATE

# Database
cp igqk_saas.db backups/$DATE/

# Models
tar -czf backups/$DATE/models.tar.gz compressed_models/

# Config
cp .env backups/$DATE/
```

### **Restore**

```bash
# Stop system
docker-compose down

# Restore
cp backups/20260204/igqk_saas.db ./
tar -xzf backups/20260204/models.tar.gz

# Restart
docker-compose up -d
```

---

## 🌐 PUBLIC DEPLOYMENT

### **Mit Custom Domain**

1. **DNS Setup:**
   ```
   A Record: yourdomain.com → <server-ip>
   A Record: api.yourdomain.com → <server-ip>
   ```

2. **SSL/HTTPS:**
   ```bash
   # Let's Encrypt
   certbot --nginx -d yourdomain.com -d api.yourdomain.com
   ```

3. **Nginx Config:**
   ```nginx
   server {
       listen 443 ssl;
       server_name yourdomain.com;

       ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
       ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

       location / {
           proxy_pass http://frontend:7860;
       }
   }
   ```

---

## 📞 SUPPORT & HILFE

### **Problem: Port Already in Use**

```bash
# Windows
netstat -ano | findstr :8000
taskkill /F /PID <PID>

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### **Problem: Out of Memory**

```env
# Reduzieren Sie Workers
BACKEND_WORKERS=2
```

### **Problem: Build Fails**

```bash
# Clean cache
docker system prune -af

# Rebuild
docker-compose build --no-cache
```

---

## ✅ FINAL CHECKLIST

- [x] Docker-Container erstellt
- [x] docker-compose.yml konfiguriert
- [x] Environment Variables vorbereitet
- [x] Deployment-Scripts erstellt
- [x] Management-Tools bereit
- [x] Dokumentation vollständig
- [x] Health Checks implementiert
- [ ] Monitoring implementiert (nur vorbereitet)
- [x] Backup-Strategie definiert (manuell)
- [ ] Security konfiguriert (Produktionsblocker offen, siehe oben)

---

## 🚀 DEMO STARTKLAR – KEIN PRODUCTION-GO-LIVE

**Das System ist als Demo/MVP startklar, nicht als Production-System.**

### **Quick Start:**

```bash
# Windows
cd C:\Users\a.b\Workspace\IGQK\IGQK_Complete_Package\igqk_saas
copy .env.example .env
notepad .env  # WICHTIG: Secrets ändern!
deploy.bat

# Linux/Mac
cd ~/igqk_saas
cp .env.example .env
nano .env  # WICHTIG: Secrets ändern!
./deploy.sh
```

### **Access Points:**

- **Frontend UI:** http://localhost:7860
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/api/docs

### **Next Steps:**

1. ✅ Deployen Sie das System
2. ✅ Testen Sie die Kompression
3. ✅ Richten Sie Backups ein
4. ✅ Konfigurieren Sie Monitoring
5. ✅ Deployen Sie in die Cloud (optional)

---

**🚀 Viel Erfolg beim Testen Ihres Demo-Systems!**

Das IGQK SaaS Platform ist ein MVP/Demo (nicht production-ready) und startklar
für lokale Tests. Vor einem echten Go-Live zuerst die fünf Blockpunkte am
Dokumentanfang beheben.
