# app_chatbot.py
import os, json
from typing import Dict, List
from dataclasses import dataclass

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from fastapi.middleware.cors import CORSMiddleware
# ========= 0) ENV & DEVICE =========
origins = [
    "http://127.0.0.1:5500",  # Live Server port
    "http://localhost:5500",
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
load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

MODEL_PATH = os.getenv("MODEL_PATH", "app/models/phobert_medchat_model.pt")
ID2LABEL_PATH = os.getenv("ID2LABEL_PATH", "app/assets/id2label.json")

TOP_K = int(os.getenv("TOP_K", "3"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

def pick_device():
    # Ưu tiên MPS trên Mac, sau đó CUDA, cuối cùng CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = pick_device()

# ========= 1) LABELS =========
with open(ID2LABEL_PATH, "r", encoding="utf-8") as f:
    id2label_raw = json.load(f)

# id2label_raw là dict {"0":"...", "1":"..."} -> chuyển thành list theo id tăng dần
id_pairs = sorted(((int(k), v) for k, v in id2label_raw.items()), key=lambda x: x[0])
ID2LABEL: List[str] = [v for _, v in id_pairs]
NUM_LABELS = len(ID2LABEL)

# ========= 2) MODEL =========
class SymptomClassifier(nn.Module):
    """
    Head đơn giản cho PhoBERT (RoBERTa-style).
    Nếu bạn đã fine-tune cùng cấu trúc này trong notebook thì state_dict sẽ khớp.
    """
    def __init__(self, num_labels: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # Dùng [CLS] token (vị trí 0)
        pooled = outputs.last_hidden_state[:, 0]
        x = self.dropout(pooled)
        logits = self.fc(x)
        return logits

# Khởi tạo tokenizer & model
TOKENIZER = AutoTokenizer.from_pretrained("vinai/phobert-base")
MODEL = SymptomClassifier(num_labels=NUM_LABELS)

# Load trọng số đã fine-tune
state = torch.load(MODEL_PATH, map_location="cpu")
# Cho phép strict=False nếu bạn có thêm/thiếu key không quan trọng (vd. layernorm stats)
missing, unexpected = MODEL.load_state_dict(state, strict=False)
if missing or unexpected:
    print("[Warn] Missing keys:", missing)
    print("[Warn] Unexpected keys:", unexpected)

MODEL.to(DEVICE)
MODEL.eval()

# ========= 3) INFERENCE UTILS =========
@dataclass
class InferConfig:
    top_k: int = TOP_K
    threshold: float = THRESHOLD
INF_CONF = InferConfig()

def infer_diseases(text: str, conf: InferConfig = INF_CONF) -> Dict[str, float]:
    enc = TOKENIZER(
        text, return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = MODEL(**enc)                     # [1, NUM_LABELS]

    # ↓↓↓ Quan trọng: tách khỏi graph, chuyển về CPU trước khi xử lý bằng Python
    probs = torch.sigmoid(logits).squeeze(0).detach().to("cpu")  # [NUM_LABELS]
    probs_list = probs.tolist()

    # Lọc theo threshold
    idxs = [i for i, p in enumerate(probs_list) if p >= conf.threshold]

    # Nếu không có nhãn vượt ngưỡng → vẫn lấy top-1 để gợi ý
    if not idxs:
        top1 = int(probs.argmax().item())
        idxs = [top1]

    # Sắp xếp giảm dần & lấy top_k
    idxs.sort(key=lambda i: probs_list[i], reverse=True)
    idxs = idxs[:conf.top_k]

    return {ID2LABEL[i]: round(float(probs_list[i]), 4) for i in idxs}


# Tối giản mapping bệnh -> khoa (MVP: chưa điền chi tiết, default tổng quát)
def recommend_department(predicted: Dict[str, float]) -> str:
    # TODO: Chặng 5 sẽ mở rộng mapping cụ thể theo bệnh
    return "Khám tổng quát"

# ========= 4) API =========
class PredictIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    predicted_diseases: Dict[str, float]
    top_k: List[str]
    threshold: float
    department: str

app = FastAPI(title="Medical Chatbot")

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "num_labels": NUM_LABELS, "top_k": TOP_K, "threshold": THRESHOLD}

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    preds = infer_diseases(inp.text)
    dept = recommend_department(preds)
    return PredictOut(
        predicted_diseases=preds,
        top_k=list(preds.keys()),
        threshold=THRESHOLD,
        department=dept
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
