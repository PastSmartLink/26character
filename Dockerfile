# ---- Build Stage ----
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Runtime Stage ----
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY . .

ENV PORT=8080
EXPOSE 8080

# Use Gunicorn with Cloud Run concurrency model
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app