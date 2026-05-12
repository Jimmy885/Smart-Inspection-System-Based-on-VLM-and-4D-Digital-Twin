# Smart-Inspection-System (智巡守衛)

This is a smart factory inspection and monitoring system based on **VLM (Vision Language Model)** and **4D Digital Twin (4DGS)** technologies. This project serves as the backend and display platform, receiving image frames transmitted via ROS from the edge device (AGX + ZED X), and integrates **RAG (Retrieval-Augmented Generation)** technology to provide precise anomaly analysis and action recommendations based on factory SOPs.

## System Architecture

1. **AGX Edge** (Not included in this repository): Responsible for controlling the ZED X camera, capturing images via ROS, and sending image frames to the API of this system.
2. **VLM API (Backend)**: Receives image frames, utilizes the Qdrant vector database for RAG retrieval, and calls Azure OpenAI (GPT-4.1-mini) for multi-modal analysis.
3. **Gradio Dashboard (Frontend)**: Provides Live Feed monitoring, analysis results display (Dashboard), and 4DGS model browsing capabilities.

---

## Directory Structure

```text
./
├── .env                # Environment variables (API Keys, Endpoints)
├── docker-compose.yml  # Container deployment configuration
├── api/                # VLM analysis backend (FastAPI)
│   ├── app.py          # Main VLM + RAG logic
│   ├── Dockerfile
│   └── requirements.txt
├── gradio/             # Monitoring dashboard frontend (Gradio)
│   ├── gradio_app.py   # Real-time monitoring interface
│   ├── Dockerfile
│   └── requirements.txt
├── data/               # Data storage
│   ├── vlm_results.db  # SQLite database (stores analysis history)
│   ├── snapshots/      # Stores the latest analyzed frames and live feeds
│   ├── docs/           # SOP documents for RAG reference (.txt)
│   └── 4DGS/           # Digital twin model files (.glb)
└── record/             # Backup and records
```

---

## Core Features

### 1. VLM Multi-modal Analysis
* **Multi-image Comparison**: Supports comparison between "Current State" and "Historical State" to determine equipment displacement, damage, or abnormal trends.
* **RAG Enhancement**: The system automatically retrieves the most relevant SOP content from `data/docs/` based on the analysis request, ensuring analysis recommendations comply with factory regulations.

### 2. Real-time Monitoring & Analysis Dashboard
* **High-Frequency Live Feed**: Refreshes the real-time stream at a high frequency to ensure uninterrupted monitoring.
* **Low-Frequency Analysis**: Sends analysis requests at scheduled intervals and updates Risk Level, Confidence, and recommended Actions.

### 3. 4DGS Digital Twin Browsing
* Integrates a 3D model browser, supporting direct viewing of factory `.glb` models converted via 4DGS (Gaussian Splatting) on the monitoring platform to achieve digital twin inspection.

### 4. Anomaly Event Management
* Automatically saves all analysis results into an SQLite database.
* Provides historical record queries; users can click to view detailed evidence, SOP reference sources, and recommended actions.

---

## Deployment Instructions

### Prerequisites
Please ensure Docker and Docker Compose are installed on your system.

### Environment Variables
Create a `.env` file in the root directory with the following content:

```env
# Azure OpenAI Settings
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=[https://your-endpoint.openai.azure.com/](https://your-endpoint.openai.azure.com/)
AZURE_OPENAI_DEPLOYMENT=gpt-4-mini
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# Qdrant & Vector DB
QDRANT_URL=http://qdrant:6333
DOC_COLLECTION=sop_docs

# System Settings
DEBUG=true
ANALYZE_QUERY="You are a senior equipment engineer..."
```

### Start the System

```bash
docker-compose up --build
```

After starting:
* **Gradio UI**: `http://localhost:7860`
* **API Docs**: `http://localhost:8000/docs`


## Collaboration Guide

* **Image Input**: The AGX edge device must convert images to Base64 format and call the `/analyze` endpoint, or send image bytes directly to the `/live_feed` endpoint.
* **SOP Updates**: Simply place new `.txt` files into `data/docs/` and call the `/ingest_dir` API to complete the knowledge base update.
