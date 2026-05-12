import os, base64, json, glob, time, logging, threading, sqlite3
from typing import List, Optional
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import datetime 

import httpx
from fastapi import FastAPI, Body, HTTPException, UploadFile, File 
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI

load_dotenv()

DEFAULT_ANALYZE_QUERY = os.getenv(
    "ANALYZE_QUERY",
    "你是資深設備工程師與工廠安全專家。你的任務是分析視覺數據並根據SOP提供專業判斷，輸出JSON {type,evidence,risk_level,actions,references,confidence}"
)

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AzureOPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://hislab-openai.openai.azure.com/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

if not AZURE_OPENAI_API_KEY:
    raise RuntimeError("Missing AZURE_OPENAI_API_KEY in .env file")

gpt_client = AzureOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)

DOC_COLLECTION = os.getenv("DOC_COLLECTION", "sop_docs")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333") 
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("smart-patrol")

DB_PATH = "/app/data/vlm_results.db"
SNAP_DIR = "/app/data/snapshots"
LATEST_CURRENT_IMG = os.path.join(SNAP_DIR, "latest_current.jpg")
LATEST_HISTORICAL_IMG = os.path.join(SNAP_DIR, "latest_historical.jpg")
LIVE_FEED_IMG = os.path.join(SNAP_DIR, "live_feed.jpg")
os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        self._create_table()
        log.info(f"Database connected at {db_path}")

    def _create_table(self):
        cursor = self.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vlm_results(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                alert_type TEXT,
                risk_level TEXT,
                status TEXT,
                evidence TEXT,
                actions TEXT,
                sop_references TEXT,
                confidence TEXT
            )
        """)
        self.connection.commit()

    def insert_result(self, timestamp, alert_type, risk_level, status, evidence, actions, sop_references, confidence):
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                "INSERT INTO vlm_results (timestamp, alert_type, risk_level, status, evidence, actions, sop_references, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (str(timestamp), str(alert_type), str(risk_level), str(status), str(evidence), str(actions), str(sop_references), str(confidence))
            )
            self.connection.commit()
            log.info(f"Successfully inserted record: {alert_type}")
        except Exception as e:
            log.error(f"Failed to insert record: {e}")

db_manager = DatabaseManager(DB_PATH)

def save_snapshot(b64_data, path):
    try:
        jpeg_b64 = normalize_to_jpeg_b64(b64_data)
        img_data = base64.b64decode(jpeg_b64)
        
        with open(path, 'wb') as f:
            f.write(img_data)
        log.info(f"Analysis snapshot updated: {path}")
    except Exception as e:
        log.error(f"Failed to save analysis snapshot: {e}")

def save_live_feed(image_bytes: bytes, path: str):
    try:
        with open(path, 'wb') as f:
            f.write(image_bytes)
    except Exception as e:
        log.error(f"Failed to save live feed snapshot: {e}")

log.info("Loading text embedding model BAAI/bge-m3 ...")
text_model = SentenceTransformer("BAAI/bge-m3")
dim = text_model.get_sentence_embedding_dimension()

log.info(f"Connecting Qdrant at {QDRANT_URL} ...")
qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

def ensure_collection():
    try:
        qdr.get_collection(DOC_COLLECTION)
    except Exception:
        log.info(f"Creating Qdrant collection: {DOC_COLLECTION} (dim={dim})")
        qdr.create_collection(
            DOC_COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )

ensure_collection()

class DocItem(BaseModel):
    id: Optional[str] = None
    text: str = Field(..., min_length=1)
    meta: Optional[dict] = None

class AnalyzeReq(BaseModel):
    query: str = Field(default=DEFAULT_ANALYZE_QUERY)
    images_b64: List[str] = Field(min_length=2, max_length=2)
    topk: int = 4

def ingest_docs_from_folder():
    ensure_collection()
    docs_dir = "/app/data/docs" 
    
    current_file_ids = set()
    paths = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
    for p in paths:
        try:
            file_id = int(os.path.splitext(os.path.basename(p))[0])
            current_file_ids.add(file_id)
        except ValueError:
            log.warning(f"File {os.path.basename(p)} does not match numeric ID format, skipping")
            continue

    try:
        existing_points, _ = qdr.scroll(
            collection_name=DOC_COLLECTION, 
            limit=10000, 
            with_payload=False, 
            with_vectors=False
        )
        existing_ids = {point.id for point in existing_points}
        log.info(f"Qdrant collection currently has {len(existing_ids)} points")
    except Exception as e:
        log.error(f"Could not fetch existing data from Qdrant: {e}")
        existing_ids = set()

    ids_to_delete = list(existing_ids - current_file_ids)
    
    if ids_to_delete:
        log.info(f"Deleting {len(ids_to_delete)} expired points from Qdrant: {ids_to_delete}")
        qdr.delete(
            collection_name=DOC_COLLECTION,
            points_selector=models.PointIdsList(points=ids_to_delete)
        )

    if not paths:
        log.warning(f"No text files found in {docs_dir}")
        return {"ingested": 0, "deleted": len(ids_to_delete)}

    texts, ids, payloads = [], [], []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            if not txt:
                continue
            file_id = int(os.path.splitext(os.path.basename(p))[0])
            texts.append(txt)
            ids.append(file_id)
            payloads.append({"text": txt, "source": os.path.basename(p)})
        except Exception as e:
            log.error(f"Failed to read file {p}: {e}")

    if not texts:
        return {"ingested": 0, "deleted": len(ids_to_delete)}

    vecs = text_model.encode(texts, normalize_embeddings=True)
    points = [
        models.PointStruct(id=ids[i], vector=vecs[i].tolist(), payload=payloads[i])
        for i in range(len(texts))
    ]
    qdr.upsert(DOC_COLLECTION, points=points)
    log.info(f"Ingested/Updated {len(points)} points from {docs_dir}")
    return {"ingested": len(points), "deleted": len(ids_to_delete)}

def normalize_to_jpeg_b64(b64_or_dataurl: str) -> str:
    if "," in b64_or_dataurl:
        b64_or_dataurl = b64_or_dataurl.split(",", 1)[1]
    raw = base64.b64decode(b64_or_dataurl)
    img = Image.open(BytesIO(raw)).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()

def call_gpt(images_b64: List[str], query: str, rag_texts: List[str]):
    current_image_b64 = images_b64[0]
    historical_image_b64 = images_b64[1]
    system_prompt = (
        "你是資深設備工程師與工廠安全專家。你的任務是分析視覺數據並根據SOP提供專業判斷。"
        "你會非常精確的描述當下的的狀態並避免不必要的描述。"
        "你的輸出必須是嚴格的 JSON 格式，包含以下欄位: "
        "{'type': '異常類型', 'evidence': '具體事證描述', 'risk_level': '風險等級 (0-5)', "
        "'actions': ['建議處理步驟'], 'references': ['參考SOP文件來源'], 'confidence': '信心指數 (0%-99%)', "
        "'rag_summary': '基於SOP文件對當前畫面的總結與判斷依據(一句話總結)'}"
    )
    user_prompt_text = (
        "作為一名資深設備工程師與工廠安全專家，請分析以下兩張圖片。\n"
        "首張是「當前狀態」，次張是「歷史狀態」。\n"
        "比較兩張圖片若是有物體狀態變化或異常有越發嚴重之趨勢,判斷這些差異是否構成潛在的異常或安全風險。\n"
        "如果兩張圖片狀態相似則不須提到差異以描述當下狀態為主\n"
        "結合下方提供的SOP文件,生成分析報告。"
    )

    content = [
        {"type": "text", "text": user_prompt_text},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{current_image_b64}"}
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{historical_image_b64}"}
        }
    ]
    if rag_texts:
        clipped = [t[:1200] for t in rag_texts]
        rag_text = "以下為SOP/手冊參考（節錄）：\n" + "\n---\n".join(clipped)
        content.append({"type": "text", "text": rag_text})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    resp = gpt_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,             
        messages=messages,
        response_format={"type": "json_object"},
    )

    text = resp.choices[0].message.content.strip()
    try:
        parsed = json.loads(text)
        return {"json": parsed, "raw_text": text}
    except Exception:
        return {"raw_text": text}

app = FastAPI(title="VLM + RAG")

@app.on_event("startup")
async def _startup_async():
    try:
        ingest_docs_from_folder()
    except Exception as e:
        log.warning(f"Initial ingestion failed: {e}")
    app.state.http = httpx.AsyncClient(timeout=30.0)

@app.on_event("shutdown")
async def _shutdown_async():
    try:
        await app.state.http.aclose()
    except Exception:
        pass

@app.get("/health")
def health():
    return {"ok": True, "collection": DOC_COLLECTION}

@app.post("/live_feed")
async def live_feed(live_image: UploadFile = File(...)):
    try:
        image_bytes = await live_image.read()
        save_live_feed(image_bytes, LIVE_FEED_IMG)
        return {"status": "live feed received"}
    except Exception as e:
        log.error(f"Process live_feed failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process live feed")

@app.post("/upsert_docs")
def upsert_docs(docs: List[DocItem]):
    ensure_collection()
    texts = [d.text for d in docs]
    vecs = text_model.encode(texts, normalize_embeddings=True)
    points = []
    for i, (d, v) in enumerate(zip(docs, vecs)):
        pid = d.id or f"{DOC_COLLECTION}_{int(time.time())}_{i}"
        payload = {"text": d.text}
        if d.meta:
            payload.update(d.meta)
        points.append(models.PointStruct(id=pid, vector=v.tolist(), payload=payload))
    qdr.upsert(DOC_COLLECTION, points=points)
    return {"count": len(points)}

@app.get("/search")
def search(q: str, k: int = 5):
    ensure_collection()
    qv = text_model.encode([q], normalize_embeddings=True)[0].tolist()
    hits = qdr.search(DOC_COLLECTION, query_vector=qv, limit=k)
    return [{"score": float(h.score), "payload": h.payload} for h in hits]

@app.post("/ingest_dir")
def ingest_dir():
    return ingest_docs_from_folder()

@app.post("/analyze")
async def analyze(req: AnalyzeReq):
    if len(req.images_b64) != 2:
        raise HTTPException(status_code=422, detail="Request must contain exactly two base64 images.")

    try:
        save_snapshot(req.images_b64[0], LATEST_CURRENT_IMG)
        save_snapshot(req.images_b64[1], LATEST_HISTORICAL_IMG)
    except Exception as e:
        log.error(f"Snapshot storage failed: {e}")

    ensure_collection()
    qv = text_model.encode([req.query], normalize_embeddings=True)[0].tolist()
    hits = qdr.search(DOC_COLLECTION, query_vector=qv, limit=max(1, req.topk))
    rag_texts = [h.payload.get("text", "") for h in hits]

    norm_images = []
    for b in req.images_b64:
        try:
            norm_images.append(normalize_to_jpeg_b64(b))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image decode error: {e}")

    t0 = time.time()

    try:
        resp = call_gpt(norm_images, req.query, rag_texts)
        parsed = resp if "json" in resp else {"raw_text": resp.get("raw_text")}
    except Exception as e:
        log.exception("Azure OpenAI call failed")
        raise HTTPException(status_code=502, detail=f"Azure OpenAI request failed: {e}")

    latency_ms = int((time.time() - t0) * 1000)

    if isinstance(parsed.get("json"), (dict, list)):
        llm_text = json.dumps(parsed["json"], ensure_ascii=False)
    elif isinstance(parsed.get("raw_text"), str):
        llm_text = parsed["raw_text"]
    else:
        llm_text = ""

    preview = parsed.get("raw_text") or (parsed.get("json") and json.dumps(parsed["json"], ensure_ascii=False))
    log.info("LLM preview: %s", (preview or ""))

    content_summary = f"Query: {req.query[:100]}..., Images: {len(req.images_b64)} imgs, RAG docs: {len(rag_texts)}"
    log.info("Calling GPT with summary: %s", content_summary)

    try:
        llm_json = parsed.get("json", {})
        alert = llm_json.get("type", "N/A")
        
        def convert_to_string(d):
            if isinstance(d, list):
                return ", ".join(str(item) for item in d)
            return str(d) if d is not None else "N/A"
        
        is_normal = (alert and alert in ["安全", "正常", "無異常"])
        
        db_manager.insert_result(
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            alert_type=alert,
            risk_level=convert_to_string(llm_json.get("risk_level")),
            status="Processed" if is_normal else "Pending",
            evidence=convert_to_string(llm_json.get("evidence")),
            actions=convert_to_string(llm_json.get("actions")),
            sop_references=convert_to_string(llm_json.get("references")),
            confidence=convert_to_string(llm_json.get("confidence"))
        )

    except Exception as e:
        log.error(f"Error occurred during database storage: {e}")

    return {
        "query": req.query,
        "rag_hits": len(hits),
        "rag_texts": [t[:200] + ("..." if len(t) > 200 else "") for t in rag_texts],
        "llm_text": llm_text,
        "result": parsed,
        "latency_ms": latency_ms
    }