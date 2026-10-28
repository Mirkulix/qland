# 🚀 IGQK v3.0 - SaaS Platform

**All-in-One ML Platform: Training + Compression + Deployment**

## 🎯 Features

### 🔨 CREATE Mode
- Train models from scratch with Quantum-optimized IGQK
- Support for all major datasets (HuggingFace, Kaggle, custom)
- Visual architecture builder
- AutoML capabilities
- One-click publishing to HuggingFace/GitHub

### 🗜️ COMPRESS Mode
- 16× compression with IGQK Quantum technology
- Support for all PyTorch models
- Multiple compression methods (Ternary, Binary, Sparse, Low-Rank)
- A/B testing framework
- Multi-cloud deployment

## 📁 Project Structure

```
igqk_saas/
├── backend/              # FastAPI backend
│   ├── api/             # API endpoints
│   ├── services/        # Business logic
│   ├── models/          # Database models
│   ├── core/            # Core IGQK integration
│   └── utils/           # Utilities
├── frontend/            # React frontend
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Pages
│   │   ├── services/    # API services
│   │   └── utils/       # Utilities
│   └── public/
├── shared/              # Shared code
├── tests/               # Tests
└── docs/                # Documentation
```

## 🚀 Quick Start

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## 🔑 Environment Variables

```bash
# Backend
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
JWT_SECRET=...
HUGGINGFACE_TOKEN=...

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 📊 Tech Stack

- **Backend:** FastAPI, PostgreSQL, Redis, Celery
- **Frontend:** Next.js, React, TailwindCSS
- **ML:** PyTorch, IGQK Core, HuggingFace
- **Infrastructure:** Docker, Kubernetes, AWS/GCP

## 🎓 Documentation

See `/docs` for detailed documentation.

## 📝 License

MIT License

## Backend starten (2026-10-26)

Das Backend braucht `huggingface-hub < 1.0`: Mit neueren Versionen verweigert `transformers` den
Import, und das Backend startet nicht. Eine eigene Umgebung, die die vorhandenen Pakete wie
`torch` mitbenutzt:

```bash
cd IGQK_Complete_Package/igqk_saas
python3 -m venv --system-site-packages .venv
.venv/bin/pip install "huggingface-hub>=0.23.2,<1.0"
cd backend && ../.venv/bin/python main.py        # Tests: ../.venv/bin/python -m pytest tests
```
