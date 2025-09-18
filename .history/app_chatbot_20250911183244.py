# app_chatbot.py
"""
Vietnamese Medical Chatbot with Conversational Flow
- Collects patient information through natural dialogue using Gemini API
- Systematically fills required medical slots 
- Calls PhoBERT model when all information is gathered
- Maps predictions to appropriate medical departments
"""

import re
import os, json
from typing import Dict, List
from dataclasses import dataclass

from fastapi import Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.db import get_db
from app.models import PatientRecord
import uuid, pandas as pd


from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Import new conversation components
from app.schemas import ChatIn, ChatOut, LLMOutput, REQUIRED_SLOTS
from app.state_store import get_state, merge_slots
from app.gemini_service import call_gemini

# ========= CONFIGURATION =========
load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# PhoBERT Model Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "app/models/phobert_medchat_model.pt")
ID2LABEL_PATH = os.getenv("ID2LABEL_PATH", "app/assets/id2label.json")

# Prediction Configuration
TOP_K = int(os.getenv("TOP_K", "3"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

# Department Mapping Configuration
DEPARTMENT_DEFAULT = os.getenv("DEPARTMENT_DEFAULT", "Khám tổng quát")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEPARTMENT_MAP_PATH = os.getenv("DEPARTMENT_MAP_PATH", os.path.join(SCRIPT_DIR, "app", "department_map.json"))

# ========= DEPARTMENT MAPPING =========
def load_department_map(path: str) -> dict:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

DEPT_MAP = load_department_map(DEPARTMENT_MAP_PATH)

# Regex rules for disease-to-department mapping
REGEX_RULES = [
    (r"(?i)ung\s*thư", "Ung bướu"),
    (r"(?i)viêm\s*gan", "Tiêu hoá – Gan mật"),
    (r"(?i)gan\s*to", "Tiêu hoá – Gan mật"),
    (r"(?i)trào\s*ngược|GERD", "Tiêu hoá"),
    (r"(?i)dạ\s*dày|ruột|loét", "Tiêu hoá"),
    (r"(?i)tiêu\s*chảy|ngộ\s*độc", "Tiêu hoá"),
    (r"(?i)hen|viêm\s*phổi|khó\s*thở|cảm\s*lạnh", "Nội hô hấp"),
    (r"(?i)covid", "Sàng lọc COVID / Truyền nhiễm"),
    (r"(?i)sốt\s*xuất\s*huyết|sốt\s*rét|thương\s*hàn|AIDS|thủy\s*đậu|truyền\s*nhiễm", "Truyền nhiễm"),
    (r"(?i)đau\s*tim|tăng\s*huyết\s*áp|cao\s*huyết\s*áp|mạch|bẩm\s*sinh\s*tim|lỗ\s*thông\s*bầu\s* dục", "Tim mạch"),
    (r"(?i)tiểu\s*đường|đái\s*tháo\s*đường|cường\s*giáp|suy\s*giáp|hạ\s*đường\s*huyết", "Nội tiết"),
    (r"(?i)thiếu\s*máu", "Huyết học"),
    (r"(?i)viêm\s*khớp|xương\s*khớp", "Cơ xương khớp"),
    (r"(?i)gãy|rách|chấn\s*thương|đầu\s*gối|mắt\s*cá", "Chấn thương chỉnh hình"),
    (r"(?i)nhiễm\s*trùng\s*đường\s*tiết\s*niệu|thận", "Thận – Tiết niệu"),
    (r"(?i)âm\s*đạo|thai\s*kỳ|sản", "Sản phụ khoa"),
    (r"(?i)mụn|vẩy\s*nến|gàu|chốc\s*lở|da\s*liễu", "Da liễu"),
    (r"(?i)đau\s*nửa\s*đầu|chóng\s*mặt|mất\s*thăng\s*bằng|thần\s*kinh", "Thần kinh"),
    (r"(?i)tràn\s*khí\s*màng\s*phổi|tắc\s*ruột|xuất\s*huyết", "Cấp cứu")
]

def recommend_department(predicted: Dict[str, float]) -> str:
    """Map predicted diseases to appropriate medical departments"""
    if not predicted:
        return DEPARTMENT_DEFAULT

    # Get the top predicted disease
    top_label = max(predicted, key=predicted.get)
    
    # Try exact match first
    if top_label in DEPT_MAP:
        return DEPT_MAP[top_label]
    
    # Try case-insensitive match
    top_label_lower = top_label.lower().strip()
    for k, v in DEPT_MAP.items():
        if k.lower().strip() == top_label_lower:
            return v

    # Try regex pattern matching
    for pattern, dept in REGEX_RULES:
        if re.search(pattern, top_label_lower):
            return dept

    # Default fallback
    return DEPARTMENT_DEFAULT

# ========= DEVICE CONFIGURATION =========
def pick_device():
    """Select the best available device for PyTorch"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = pick_device()

# ========= DISEASE LABEL LOADING =========
with open(ID2LABEL_PATH, "r", encoding="utf-8") as f:
    id2label_raw = json.load(f)

# Convert id2label dict to sorted list
id_pairs = sorted(((int(k), v) for k, v in id2label_raw.items()), key=lambda x: x[0])
ID2LABEL: List[str] = [v for _, v in id_pairs]
NUM_LABELS = len(ID2LABEL)

# ========= PHOBERT MODEL =========
class SymptomClassifier(nn.Module):
    """PhoBERT-based classifier for Vietnamese medical symptoms"""
    
    def __init__(self, num_labels: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0]
        x = self.dropout(pooled)
        logits = self.fc(x)
        return logits

# Initialize tokenizer and model
TOKENIZER = AutoTokenizer.from_pretrained("vinai/phobert-base")
MODEL = SymptomClassifier(num_labels=NUM_LABELS)

# Load fine-tuned weights
state = torch.load(MODEL_PATH, map_location="cpu")
missing, unexpected = MODEL.load_state_dict(state, strict=False)
if missing or unexpected:
    print("[Warning] Missing/unexpected keys in model state dict")

MODEL.to(DEVICE)
MODEL.eval()

# ========= INFERENCE FUNCTIONS =========
@dataclass
class InferConfig:
    top_k: int = TOP_K
    threshold: float = THRESHOLD

INF_CONF = InferConfig()

def infer_diseases(text: str, conf: InferConfig = INF_CONF) -> Dict[str, float]:
    """Predict diseases from Vietnamese symptom text using PhoBERT"""
    enc = TOKENIZER(
        text, return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = MODEL(**enc)

    # Convert logits to probabilities and move to CPU
    probs = torch.sigmoid(logits).squeeze(0).detach().to("cpu")
    probs_list = probs.tolist()

    # Filter by threshold
    idxs = [i for i, p in enumerate(probs_list) if p >= conf.threshold]

    # If no predictions above threshold, take top-1
    if not idxs:
        top1 = int(probs.argmax().item())
        idxs = [top1]

    # Sort by probability descending and take top_k
    idxs.sort(key=lambda i: probs_list[i], reverse=True)
    idxs = idxs[:conf.top_k]

    return {ID2LABEL[i]: round(float(probs_list[i]), 4) for i in idxs}

# ========= FASTAPI APPLICATION =========
app = FastAPI(title="Medical Chatbot - Vietnamese PhoBERT System")

# CORS Configuration
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500", 
    "http://127.0.0.1:5501",
    "http://localhost:5501",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= API MODELS =========
class PredictIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    predicted_diseases: Dict[str, float]
    top_k: List[str]
    threshold: float
    department: str

# ========= API ENDPOINTS =========
@app.get("/health")
def health():
    """System health check"""
    return {
        "status": "ok", 
        "device": str(DEVICE), 
        "num_labels": NUM_LABELS, 
        "top_k": TOP_K, 
        "threshold": THRESHOLD
    }

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    """Direct disease prediction from symptoms (legacy endpoint)"""
    preds = infer_diseases(inp.text)
    dept = recommend_department(preds)
    return PredictOut(
        predicted_diseases=preds,
        top_k=list(preds.keys()),
        threshold=THRESHOLD,
        department=dept
    )

@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn):
    """
    Main conversational endpoint:
    - Uses Gemini API to collect patient information through dialogue
    - Systematically fills required medical slots
    - Calls PhoBERT model when all information is gathered
    - Returns appropriate medical department recommendations
    """
    user_id = body.user_id
    message = body.message

    # 1) Get current conversation state
    state = get_state(user_id)

    # 2) Call Gemini LLM with structured JSON schema
    llm_out = call_gemini(message, state)
    parsed = LLMOutput(**llm_out)

    # 3) Merge newly extracted slots into state
    filled = merge_slots(user_id, parsed.slots_extracted)

    # 4) Calculate missing slots
    missing = [s for s in REQUIRED_SLOTS if not filled.get(s)]

    # 5) Decision branch: call PhoBERT if all slots filled
    if parsed.next_action == "call_phobert" and not missing:
        text_for_model = filled.get("symptoms") or message
        preds = infer_diseases(text_for_model)
        dept = recommend_department(preds)
        assist_msg = (
            f"Kết quả phân tích: {', '.join([f'{k} ({v})' for k, v in preds.items()])}. "
            f"Khoa khám gợi ý: {dept}."
        )
        next_action = "call_phobert"
    else:
        assist_msg = parsed.assistant_message
        next_action = "ask_for_missing_slots"

    return ChatOut(
        assistant_message=assist_msg,
        filled_slots=filled,
        missing_slots=missing,
        next_action=next_action,
        debug={"llm_next_action": parsed.next_action}
    )

# ========= MAIN =========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)