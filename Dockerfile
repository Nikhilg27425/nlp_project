FROM python:3.11-slim

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install Python deps first (cached layer)
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-api.txt

# copy source
COPY python_autocomplete/ ./python_autocomplete/
COPY webapp/ ./webapp/

# model bundle (optional — mount at runtime if large)
# COPY webapp/model_bundle.pt ./webapp/model_bundle.pt

EXPOSE 5001

# health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5001/health || exit 1

CMD ["uvicorn", "webapp.api:app", "--host", "0.0.0.0", "--port", "5001", \
     "--workers", "1", "--log-level", "info"]
