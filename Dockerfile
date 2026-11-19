FROM python:3.11.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY backend /app/backend
COPY src /app/src
COPY run.py .
COPY prepare_runtime.py .

ENV FLASK_ENV=production

EXPOSE 8080

CMD ["gunicorn", "run:app", "--bind=0.0.0.0:8080", "--workers=1", "--timeout=600"]