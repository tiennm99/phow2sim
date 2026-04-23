FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# MODEL_URL + MODEL_PATH + credentials are injected at runtime via
# docker-compose env/.env (see docker-compose.yml). No defaults here —
# PhoW2V's license forbids public redistribution, so every deployment
# must point at its own private mirror (typically Nextcloud WebDAV).
ENV MODEL_PATH=/data/phow2v/word2vec_vi_words_300dims.txt \
    PORT=8000

EXPOSE 8000

# First boot downloads ~1.2GB then parses ~60s; later boots use the
# cached .bin and only need ~10s. start-period accommodates both.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
