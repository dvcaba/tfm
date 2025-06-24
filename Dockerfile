# 1) Builder stage: Genera los wheels de tus dependencias
FROM python:3.12-slim AS builder
WORKDIR /wheels

# Instala dependencias del sistema necesarias para compilar (puede variar según tus paquetes)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Genera todos los wheels en /wheels
RUN pip wheel --no-cache-dir --disable-pip-version-check -r requirements.txt


# 2) Runtime stage: solo instala los wheels precompilados y copia tu app
FROM python:3.12-slim
WORKDIR /app

# Copia e instala los wheels generados
COPY --from=builder /wheels/*.whl /wheels/
RUN pip install --no-cache-dir --disable-pip-version-check /wheels/*.whl

# Copia el código y artefactos
COPY main.py requirements.txt ./
COPY agent/ ./agent/
COPY results/ ./results/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
