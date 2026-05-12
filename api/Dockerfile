FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ffmpeg libgl1 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install -r requirements.txt
RUN pip install --default-timeout=300 -r requirements.txt
RUN pip install python-multipart
COPY app.py .
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["python","-m","debugpy","--listen","0.0.0.0:5678","-m","uvicorn","app:app","--host","0.0.0.0","--port","8000","--reload"]