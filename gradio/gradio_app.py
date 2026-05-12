import os, requests, io, time, json, sqlite3, base64, sys, glob, datetime, threading
import pandas as pd
import cv2
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from openai import AzureOpenAI

API_ENDPOINT = os.getenv("ANALYZE_ENDPOINT", "http://api:8000/analyze") 
QUERY = os.getenv("ANALYZE_QUERY", "你是工廠監控機器人...")
TOPK = int(os.getenv("RAG_TOPK", "1"))
TIMER = int(os.getenv("IDLE_SEND_SEC", "5")) 
LIVE_FEED_INTERVAL = 1.0 

snapshot_path = "/app/data/snapshots"
db_path = "/app/data/vlm_results.db"

LATEST_CURRENT_IMG = os.path.join(snapshot_path, "latest_current.jpg") 
LIVE_FEED_IMG = os.path.join(snapshot_path, "live_feed.jpg") 
os.makedirs(snapshot_path, exist_ok=True) 

DEFAULT_FONT = ImageFont.load_default()

FOURDGS_PATHS = ["/app/4DGS"]

last_frame = None
alert = "正常"
sop_text = ""
cap = None
timestamp = ""
current_expanded_row = None
expanded_states = {}
last_click_time = 0
DEBOUNCE_INTERVAL = 0.5
normal_frame = True
last_abnormal_image_path = None
last_abnormal_analysis = None
has_previous_abnormal = False
last_processed_image_path = None

class FourDGSManager:
    def __init__(self, search_paths):
        self.search_paths = search_paths
        self.available_models = {}
        self._scan_for_models()
    
    def _scan_for_models(self):
        print("Scanning for 4DGS model files...")
        for path in self.search_paths:
            if os.path.exists(path):
                print(f"Scanning path: {path}")
                glb_pattern = os.path.join(path, "**/*.glb")
                glb_files = glob.glob(glb_pattern, recursive=True)
                for glb_file in glb_files:
                    display_name = os.path.splitext(os.path.basename(glb_file))[0]
                    if display_name in self.available_models:
                        rel_path = os.path.relpath(glb_file, path)
                        display_name = f"{display_name} ({rel_path})"
                    self.available_models[display_name] = glb_file
                    print(f"Model found: {display_name}")
            else:
                print(f"Path does not exist: {path}")
        print(f"Total models found: {len(self.available_models)}")
    
    def get_model_list(self):
        return sorted(list(self.available_models.keys()))
    
    def get_model_path(self, display_name):
        return self.available_models.get(display_name)
    
    def get_default_model(self):
        if self.available_models:
            return list(self.available_models.values())[0]
        return None

fourdgs_manager = FourDGSManager(FOURDGS_PATHS)

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.connection = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        self._create_table()
    
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
        print("Gradio DBManager: insert_result (Read-only mode, skipping write)")
        pass
    
    def get_history(self, raw=False):
        cursor = self.connection.cursor()
        if raw:
            cursor.execute("SELECT evidence, actions, sop_references FROM vlm_results ORDER BY id DESC")
            rows = cursor.fetchall()
            return [dict(zip(["證據", "行動建議", "SOP參考"], row)) for row in rows]
        else:
            cursor.execute("SELECT alert_type, risk_level, timestamp, status FROM vlm_results ORDER BY id DESC")
            rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=["異常類型", "風險", "時間", "狀態"])

    def get_latest_result(self_):
        cursor = self_.connection.cursor()
        cursor.execute("""
            SELECT alert_type, risk_level, confidence, actions, evidence, sop_references 
            FROM vlm_results 
            ORDER BY id DESC LIMIT 1
        """)
        row = cursor.fetchone()
        if row:
            keys = ["alert_type", "risk_level", "confidence", "actions", "evidence", "sop_references"]
            return dict(zip(keys, row))
        return None 

    def close(self):
        self.connection.close()

db_manager = DatabaseManager(db_path)

class APIServiceManager:
    def __init__(self, api_endpoint):
        self.api_endpoint = api_endpoint
        self.health_url = api_endpoint.replace("/analyze", "/health")
        self.max_retries = 30
        self.retry_delay = 3
        self.initial_max_wait = 60
    
    def check_api_health(self):
        try:
            response = requests.get(self.health_url, timeout=10)
            if response.status_code == 200:
                return True, f"API Service Healthy: {response.json()}"
            return False, f"API Service Error: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"Connection failed: {e}"
    
    def wait_for_api_ready(self):
        print("Waiting for API service to start...")
        start_time = time.time()
        for attempt in range(self.max_retries):
            if time.time() - start_time > self.initial_max_wait:
                print(f"Max wait time of {self.initial_max_wait}s exceeded")
                return False, "Timeout"
            health_ok, health_msg = self.check_api_health()
            if health_ok:
                print(f"API ready! Total wait: {time.time() - start_time:.1f}s")
                return True, health_msg
            print(f"Attempt {attempt + 1}/{self.max_retries} - {health_msg}")
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        return False, "API not ready"
    
api_manager = APIServiceManager(API_ENDPOINT)

def load_image_from_path(path: str, default_text: str = "Waiting for image...", show_error: bool = False):
    if os.path.exists(path):
        try:
            with Image.open(path) as img:
                return img.copy() 
        except Exception as e:
            error_msg = f"File error: {os.path.basename(path)} - {e}"
            print(error_msg)
            img = Image.new('RGB', (640, 480), color = (50, 50, 50)) 
            d = ImageDraw.Draw(img)
            d.text((10,10), default_text, fill=(255, 255, 255), font=DEFAULT_FONT)
            d.text((10,40), error_msg, fill=(255, 100, 100), font=DEFAULT_FONT)
            return img 
    
    img = Image.new('RGB', (640, 480), color = (50, 50, 50))
    d = ImageDraw.Draw(img)
    d.text((10,10), default_text, fill=(255, 255, 255), font=DEFAULT_FONT)
    d.text((10,40), f"Path not found: {os.path.basename(path)}", fill=(200, 200, 200), font=DEFAULT_FONT)
    return img 

def update_analysis_panel():
    analyzed_img = load_image_from_path(LATEST_CURRENT_IMG, "Waiting for AGX Analysis image...")
    latest_result = db_manager.get_latest_result()
    
    if latest_result:
        alert = latest_result.get("alert_type", "N/A")
        sop = latest_result.get("actions", "N/A")
        risk = latest_result.get("risk_level", "N/A")
        confidence = latest_result.get("confidence", "N/A")
        evidence = latest_result.get("evidence", "N/A")
        references = latest_result.get("sop_references", "N/A")
        rag_summary = f"SOP References: {references}" if references != "N/A" else "No SOP available"
    else:
        alert, risk, confidence, sop, evidence, references, rag_summary = ("Waiting",) * 7

    alert_html = f'<div class="alert-content">{alert}</div>'
    risk_html = f'<div class="risk">{risk}</div>'
    confidence_html = f'<div class="confidence-score">{confidence}</div>'

    return (analyzed_img, alert_html, sop, risk_html, confidence_html, evidence, references, rag_summary)

def update_live_feed():
    live_img = load_image_from_path(LIVE_FEED_IMG, "Waiting for AGX Live image...")
    return live_img

def load_4dgs_model(model_name):
    if not model_name:
        return None
    model_path = fourdgs_manager.get_model_path(model_name)
    if model_path and os.path.exists(model_path):
        print(f"Loading 4DGS Model: {model_name}")
        return model_path
    print(f"Model file not found: {model_path}")
    return None

def refresh_history():
    return db_manager.get_history(raw=False)

def on_row_select(evt: gr.SelectData):
    global current_expanded_row, expanded_states, last_click_time
    current_time = time.time()
    if current_time - last_click_time < DEBOUNCE_INTERVAL:
        if current_expanded_row is not None and expanded_states.get(current_expanded_row, False):
            detail_data = db_manager.get_history(raw=True)
            if current_expanded_row < len(detail_data):
                detail_record = detail_data[current_expanded_row]
                return (detail_record.get("證據", "N/A"), detail_record.get("行動建議", "N/A"),
                        detail_record.get("SOP參考", "N/A"), gr.update(visible=True))
        return "", "", "", gr.update(visible=False)
    
    last_click_time = current_time
    row_index = evt.index[0]
    
    if row_index == current_expanded_row:
        expanded_states[row_index] = not expanded_states.get(row_index, False)
        if not expanded_states[row_index]:
             current_expanded_row = None
    else:
        if current_expanded_row is not None:
            expanded_states[current_expanded_row] = False
        expanded_states[row_index] = True
        current_expanded_row = row_index
    
    if expanded_states.get(row_index, False):
        try:
            detail_data = db_manager.get_history(raw=True)
            if row_index < len(detail_data):
                detail_record = detail_data[row_index]
                return (detail_record.get("證據", "N/A"), detail_record.get("行動建議", "N/A"),
                        detail_record.get("SOP參考", "N/A"), gr.update(visible=True))
            return "No data", "No data", "No data", gr.update(visible=True)
        except Exception as e:
            print(f"Error loading details: {e}")
            return "Error", "Error", "Error", gr.update(visible=True)
    
    return "", "", "", gr.update(visible=False)

custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap');
    .gradio-container { background-color: #f8fafc; font-family: 'Noto Sans TC', sans-serif; }
    .gradio-container, .gradio-container * { color: #374151 !important; }
    .gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container h4, .gradio-container .gr-markdown, .gradio-container .gr-markdown p { color: #1e293b !important; }
    .image-container { background-color: #000000; border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 1.5rem; }
    .status-card { background-color: white; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); margin-bottom: 1rem; border: 1px solid #e2e8f0; }
    .status-title { font-size: 1.5rem !important; font-weight: 600 !important; color: #64748b !important; margin-bottom: 0.5rem !important; display: block !important; }
    .action-box { background-color: #fffbeb; border-left: 4px solid #f59e0b; border-radius: 0 0.5rem 0.5rem 0; padding: 1.25rem; margin-top: 1rem; }
    .sop-output-text, .sop-output-text * { color: #c2410c !important; font-size: 1.5rem; font-weight: 500 !important; }
    .detail-card { background-color: #DDDDDD; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #0ea5e9; height: 100%; }
    .detail-card .detail-textbox textarea { background-color: white; color: #475569; font-size: 1.5rem; line-height: 1.5; border: none; box-shadow: none; resize: none; padding: 0.5rem; min-height: 80px; }
    .risk{ color: #f97316 !important; font-size: 2.5rem !important; font-weight: 700 !important; }
    .confidence-score { color: #0284c7 !important; font-size: 2.5rem !important; font-weight: 700 !important; }
    .alert-content { color: #1e293b !important; font-size: 2.5rem !important; font-weight: 700 !important; }
    .anomaly-management-tab { background-color: #f8fafc; padding: 1rem; border-radius: 0.5rem; }
    .anomaly-management-table { background-color: white; border-radius: 0.5rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); overflow: hidden; margin-bottom: 1.5rem; }
    .anomaly-management-table thead { background-color: white !important; border-bottom: 2px solid #e2e8f0 !important; }
    .anomaly-management-table th { background-color: white !important; color: #374151 !important; font-weight: 600 !important; font-size: 1.5rem !important; padding: 0.75rem 1rem !important; text-align: left !important; border: none !important; }
    .anomaly-management-table td { background-color: white !important; color: #374151 !important; font-size: 1.5rem !important; padding: 0.75rem 1rem !important; border: none !important; border-bottom: 1px solid #f1f5f9 !important; }
    .anomaly-management-table tbody tr:hover { background-color: #f8fafc !important; cursor: pointer; }
    @media (max-width: 768px) { .detail-box { margin-bottom: 1rem !important; } }
    .alert-content, .risk, .confidence-score { opacity: 1 !important; transition: none !important; }
    .sop-output-text, .sop-output-text * { opacity: 1 !important; transition: none !important; color: #c2410c !important; }
"""

def create_interface():
    with gr.Blocks(title="智巡守衛即時監控平台", theme=gr.themes.Soft(primary_hue="blue"), css=custom_css) as demo:
        gr.Markdown("# 🛡️ 智巡守衛即時監控平台")
        with gr.Tab("🎦 即時監控") as realtime_tab:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 即時影像 (Live Feed)")
                    with gr.Column(elem_classes="image-container"):
                        img_live = gr.Image(label="", interactive=False, value=None, show_label=False, height=400)
                with gr.Column():
                    gr.Markdown("## 當前分析幀 (Analyzed Frame)")
                    with gr.Column(elem_classes="image-container"):
                        img_analyze = gr.Image(label="", interactive=False, value=None, show_label=False, height=400)
            
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row(elem_classes="status-card"):
                        with gr.Column(scale=1):
                            gr.Markdown("# 關鍵狀態", elem_classes="status-title")
                            with gr.Column():
                                gr.Markdown("## 異常類型")
                                alert_output = gr.Markdown("分析中", elem_classes="alert-content")
                            with gr.Column():
                                gr.Markdown("## 風議等級")
                                risk_level_output = gr.Markdown("N/A", elem_classes="risk")
                            with gr.Column():
                                gr.Markdown("## 信心分數")
                                confidence_output = gr.Markdown("分析中", elem_classes="confidence-score")
                        with gr.Column(scale=2):
                            with gr.Column(elem_classes="action-box"):
                                gr.Markdown("## 建議行動")
                                sop_output = gr.Markdown("分析中", elem_classes="sop-output-text")
            with gr.Accordion(" 📋 顯示詳細資訊", open=False):
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        with gr.Column(elem_classes="detail-card"):
                            gr.Markdown("## 🔍 詳細證據")
                            evidence_output = gr.Textbox(lines=4, interactive=False, placeholder="Waiting for results...", show_label=False, elem_classes="detail-textbox")
                    with gr.Column(scale=1, min_width=300):
                        with gr.Column(elem_classes="detail-card"):
                            gr.Markdown("## 📚 SOP 參考")
                            references_output = gr.Textbox(lines=4, interactive=False, placeholder="Waiting for results...", show_label=False, elem_classes="detail-textbox")
                    with gr.Column(scale=1, min_width=300):
                        with gr.Column(elem_classes="detail-card"):
                            gr.Markdown("## 📖 RAG 檢索摘要")
                            rag_summary_output = gr.Textbox(lines=4, interactive=False, placeholder="Waiting for results...", show_label=False, elem_classes="detail-textbox")
            
            live_feed_timer = gr.Timer(LIVE_FEED_INTERVAL, active=False)
            live_feed_timer.tick(fn=update_live_feed, inputs=None, outputs=[img_live], queue=True)
            
            analysis_timer = gr.Timer(TIMER, active=False) 
            analysis_timer.tick(fn=update_analysis_panel, inputs=None, outputs=[img_analyze, alert_output, sop_output, risk_level_output, confidence_output, evidence_output, references_output, rag_summary_output], queue=True)
        
        with gr.Tab("🗺️ 4DGS 瀏覽") as model_tab:
            gr.Markdown("## 🎮 4DGS 模型瀏覽器")
            available_models = fourdgs_manager.get_model_list()
            default_model = available_models[0] if available_models else None
            with gr.Row():
                model_viewer = gr.Model3D(value=fourdgs_manager.get_default_model(), clear_color=[0.2, 0.2, 0.2, 1.0], label="👁️ 3D 模型預覽", height=500)
            with gr.Row():
                model_dropdown = gr.Dropdown(choices=available_models, value=default_model, label="📁 選擇 3D 模型", interactive=True)
            model_dropdown.change(fn=load_4dgs_model, inputs=model_dropdown, outputs=model_viewer, queue=False)
        
        with gr.Tab("⚠️ 異常事件管理", elem_classes="anomaly-management-tab") as history_tab:
            gr.Markdown("## 📊 歷史事件記錄")
            history_box = gr.DataFrame(headers=["異常類型", "風險", "時間", "狀態"], wrap=True, scale=1, datatype=["str", "str", "str", "str"], value=db_manager.get_history(raw=False), elem_classes="anomaly-management-table")
            with gr.Column(visible=False) as full_details:
                gr.Markdown("## 📋 事件詳細資訊")
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Column(elem_classes="detail-card"):
                            gr.Markdown("## 🔍 詳細證據")
                            full_evidence = gr.Textbox(lines=5, interactive=False, show_label=False, elem_classes="detail-textbox")
                    with gr.Column(scale=1):
                        with gr.Column(elem_classes="detail-card"):
                            gr.Markdown("## 🧭 行動建議")
                            full_actions = gr.Textbox(lines=5, interactive=False, show_label=False, elem_classes="detail-textbox")
                    with gr.Column(scale=1):
                        with gr.Column(elem_classes="detail-card"):
                            gr.Markdown("## 📚 SOP 參考")
                            full_sop = gr.Textbox(lines=5, interactive=False, show_label=False, elem_classes="detail-textbox")
            history_refresh_timer = gr.Timer(10)
            history_refresh_timer.tick(fn=refresh_history, inputs=None, outputs=[history_box], queue=True)
            demo.load(fn=refresh_history, inputs=[], outputs=history_box)
            history_box.select(fn=on_row_select, inputs=None, outputs=[full_evidence, full_actions, full_sop, full_details], queue=False)

        def start_timer():
            print("Interface loaded, starting timers")
            return gr.Timer(active=True), gr.Timer(active=True)
        demo.load(fn=start_timer, inputs=None, outputs=[live_feed_timer, analysis_timer])
    
    return demo

def main():
    print("Launching Guard UI...")
    api_ready, api_message = api_manager.wait_for_api_ready()
    
    if not api_ready:
        print(f"Warning: API Service not ready: {api_message}")
    else:
        print("API service connection established")

    try:
        demo = create_interface()
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True, show_error=True, quiet=False)
    except Exception as e:
        print(f"Server launch failed: {e}")
    finally:
        db_manager.close()

if __name__ == "__main__":
    main()