FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && groupadd -r appgroup && useradd -r -g appgroup appuser \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN pip install --no-cache-dir  --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY config.yaml /app/
COPY src/ /app/src/
copy app.py /app/

RUN chown -R appuser:appgroup /app

USSER appuser

EXPOSE 8000

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
