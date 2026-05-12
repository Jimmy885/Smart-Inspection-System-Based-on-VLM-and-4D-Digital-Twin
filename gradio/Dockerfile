FROM python:3.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .


RUN pip install -r requirements.txt

COPY gradio_app.py .

EXPOSE 7860

CMD ["python", "-u", "gradio_app.py"]