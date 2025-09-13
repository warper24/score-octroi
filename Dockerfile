# Stage 1: build frontend
FROM node:20-alpine AS ui
WORKDIR /web
COPY web/package.json web/vite.config.js ./
RUN npm ci
COPY web/ ./ 
RUN npm run build

# Stage 2: API
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code + artefacts + front statique
COPY service ./service
COPY --from=ui /web/dist ./service/static

EXPOSE 8000
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]