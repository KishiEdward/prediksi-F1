# Image API prediksi pit stop F1 (CPU-only, netral platform)
FROM python:3.11-slim

WORKDIR /app

# Dependensi sistem minimal untuk pandas/pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# PyTorch CPU-only diinstal dari index khusus agar image tetap kecil
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
# Sertakan dataset terproses + artifact model agar API langsung siap replay.
# (Untuk deploy dengan disk persisten, keduanya bisa di-mount sebagai volume.)
COPY data/processed ./data/processed
COPY data/raw/jolpica_races.parquet ./data/raw/jolpica_races.parquet
COPY data/raw/jolpica_results.parquet ./data/raw/jolpica_results.parquet
COPY data/raw/jolpica_constructors.parquet ./data/raw/jolpica_constructors.parquet
COPY models ./models

ENV PYTHONUNBUFFERED=1 F1_API_PORT=8000
EXPOSE 8000

CMD ["python", "-m", "app.main"]
