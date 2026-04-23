FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Defaults point at PhoW2V word-300d from VinAI's public mirror.
# Override MODEL_URL/MODEL_PATH to switch variants (syllables, 100d).
ENV MODEL_URL=https://public.vinai.io/word2vec_vi_words_300dims.zip \
    MODEL_PATH=/data/phow2v/word2vec_vi_words_300dims.txt \
    MODEL_VARIANT=word \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
