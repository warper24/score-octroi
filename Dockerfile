# Stage 1: build frontend
FROM node:20-alpine AS ui
WORKDIR /web
COPY web/package*.json ./
RUN npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build

# Stage 2: API FastAPI + artefacts
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1

# build-essential utile pour xgboost
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code + artefacts
COPY score_oc ./score_oc
COPY service ./service

# Front statique servi par FastAPI
COPY --from=ui /web/dist ./service/static

EXPOSE 8000
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]