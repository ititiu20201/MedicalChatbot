# app_chatbot.py
"""
Vietnamese Medical Chatbot with Conversational Flow
- Collects patient information through natural dialogue using Gemini API
- Systematically fills required medical slots 
- Calls PhoBERT model when all information is gathered
- Maps predictions to appropriate medical departments
- Stores patient records in database with analytics
"""

# ========= IMPORTS =========
import re
import os
import json
import uuid
from typing import Dict, List
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from sqlalchemy.orm import Session
from app.db import get_db
from app.models import PatientRecord

# Import conversation components
from app.schemas import ChatIn, ChatOut, LLMOutput, REQUIRED_SLOTS
from app.state_store import merge_slots, get_slots, get_off_topic_count, increment_off_topic_count, update_fields
from app.gemini_service import call_gemini

# ========= CONFIGURATION =========
load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8003"))

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

    def forward(self, input_ids, attention_mask):
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
    # Filter out token_type_ids which PhoBERT model doesn't expect
    enc = {k: v.to(DEVICE) for k, v in enc.items() if k != 'token_type_ids'}

    with torch.no_grad():
        logits = MODEL(**enc)

    # Convert logits to probabilities and move to CPU
    probs = torch.sigmoid(logits).squeeze(0).detach().to("cpu")
    probs_list = probs.tolist()

    # Filter by threshold
    idxs = [i for i, p in enumerate(probs_list) if p >= conf.threshold]

    # If no predictions above threshold, take top-k anyway
    if not idxs:
        # Get all indices sorted by probability
        all_idxs = list(range(len(probs_list)))
        all_idxs.sort(key=lambda i: probs_list[i], reverse=True)
        idxs = all_idxs[:conf.top_k]
    else:
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
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:8080",
    "http://localhost:8080",
    "null",  # For file:// protocol
    "*",     # Allow all origins for development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Disable credentials when using wildcard
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
def chat(body: ChatIn, db: Session = Depends(get_db)):
    """
    Main conversational endpoint with database storage:
    - Uses Gemini API to collect patient information through dialogue
    - Systematically fills required medical slots
    - Calls PhoBERT model when all information is gathered
    - Returns appropriate medical department recommendations
    - Saves completed consultations to database
    """
    user_id = body.user_id
    message = body.message

    # 1) Get current slots and off-topic count
    slots = get_slots(user_id)
    off_topic_count = get_off_topic_count(user_id)

    # 2) Call Gemini LLM with structured JSON schema and off-topic context
    llm_out = call_gemini(message, slots, off_topic_count)
    parsed = LLMOutput(**llm_out)

    # 3) Handle field updates first (corrections/modifications)
    if parsed.field_updates:
        filled = update_fields(user_id, parsed.field_updates)
    else:
        # 4) Merge newly extracted slots into state (only if not updating)
        filled = merge_slots(user_id, parsed.slots_extracted)

    # 5) Calculate missing slots
    missing = [s for s in REQUIRED_SLOTS if not filled.get(s)]

    # 5) Handle off-topic responses first
    if parsed.next_action == "off_topic_response" or parsed.is_off_topic:
        # Increment off-topic counter
        new_count = increment_off_topic_count(user_id)

        if new_count >= 2:
            # Limit reached - use response without additional guidance
            assist_msg = parsed.assistant_message
            next_action = "guide_back_to_medical"
        else:
            # Still within limit
            assist_msg = parsed.assistant_message
            next_action = "off_topic_response"

    # 6) Handle information recall queries
    elif parsed.next_action == "information_recall":
        assist_msg = parsed.assistant_message
        next_action = "information_recall"

    # 6.5) Handle information update confirmations
    elif parsed.next_action == "information_updated":
        assist_msg = parsed.assistant_message
        next_action = "information_updated"

        # Save field updates to database immediately when all information is complete
        if not missing:
            existing_record = db.query(PatientRecord).filter(PatientRecord.user_id == user_id).first()

            if existing_record:
                # Update existing record with latest field updates
                existing_record.patient_name = filled.get("name", "")
                existing_record.patient_phone = filled.get("phone_number", "")
                existing_record.patient_age = filled.get("age", "")
                existing_record.patient_gender = filled.get("gender", "")
                existing_record.symptoms = filled.get("symptoms", "")
                existing_record.onset = filled.get("onset", "")
                existing_record.allergies = filled.get("allergies", "")
                existing_record.current_medications = filled.get("current_medications", "")
                existing_record.pain_scale = filled.get("pain_scale", "")
                existing_record.chat = json.dumps({"filled": filled, "missing": missing}, ensure_ascii=False)
                db.commit()
                print(f"✅ Updated existing patient record for user {user_id} with field changes")
            else:
                print(f"⚠️ No existing record found for user {user_id} during field update")

    # 7) Decision branch: handle different conversation states
    elif parsed.next_action == "call_phobert" and not missing:
        text_for_model = filled.get("symptoms") or message
        preds = infer_diseases(text_for_model)
        dept = recommend_department(preds)

        # Upsert patient record to database (update if exists, insert if new)
        existing_record = db.query(PatientRecord).filter(PatientRecord.user_id == user_id).first()

        if existing_record:
            # Update existing record with latest information
            existing_record.patient_name = filled.get("name", "")
            existing_record.patient_phone = filled.get("phone_number", "")
            existing_record.patient_age = filled.get("age", "")
            existing_record.patient_gender = filled.get("gender", "")
            existing_record.symptoms = filled.get("symptoms", "")
            existing_record.onset = filled.get("onset", "")
            existing_record.allergies = filled.get("allergies", "")
            existing_record.current_medications = filled.get("current_medications", "")
            existing_record.pain_scale = filled.get("pain_scale", "")
            existing_record.predicted_diseases = json.dumps(preds, ensure_ascii=False)
            existing_record.recommended_department = dept
            existing_record.chat = json.dumps({"filled": filled, "missing": missing}, ensure_ascii=False)
            # Keep original record_id and created_at, but update timestamp implicitly
        else:
            # Create new record for first-time consultation
            record = PatientRecord(
                record_id=str(uuid.uuid4()),
                user_id=user_id,
                # Patient Information
                patient_name=filled.get("name", ""),
                patient_phone=filled.get("phone_number", ""),
                patient_age=filled.get("age", ""),
                patient_gender=filled.get("gender", ""),
                # Medical Information
                symptoms=filled.get("symptoms", ""),
                onset=filled.get("onset", ""),
                allergies=filled.get("allergies", ""),
                current_medications=filled.get("current_medications", ""),
                pain_scale=filled.get("pain_scale", ""),
                # Results
                predicted_diseases=json.dumps(preds, ensure_ascii=False),
                recommended_department=dept,
                # System
                chat=json.dumps({"filled": filled, "missing": missing}, ensure_ascii=False),
            )
            db.add(record)

        db.commit()

        # After diagnosis, ask final confirmation
        assist_msg = (
            f"🩺 **Kết quả phân tích:**\n"
            f"• Dự đoán bệnh: {', '.join([f'{k} ({(v*100):.1f}%)' for k, v in preds.items()])}\n"
            f"• Khoa khám gợi ý: **{dept}**\n\n"
            f"📋 Bạn có thể in hồ sơ tư vấn để mang đến bệnh viện.\n\n"
            f"❓ Bạn có muốn hỏi thêm điều gì khác không? Nếu không, hãy nhập 'kết thúc' để hoàn tất phiên tư vấn."
        )
        next_action = "final_confirmation"
    elif parsed.next_action == "session_complete" or message.lower().strip() in ['kết thúc', 'ket thuc', 'end', 'finish', 'hoàn tất', 'hoan tat']:
        assist_msg = (
            f"🙏 **Cảm ơn bạn đã sử dụng dịch vụ tư vấn y tế AI!**\n\n"
            f"📋 Hồ sơ tư vấn của bạn đã được lưu và sẵn sàng để in.\n"
            f"🏥 Vui lòng đến bệnh viện để được thăm khám chính xác bởi bác sĩ chuyên khoa.\n\n"
            f"🔄 **Để bắt đầu phiên tư vấn mới, vui lòng nhấn nút 'Phiên mới' để làm mới cuộc trò chuyện.**\n\n"
            f"⚠️ *Lưu ý: Thông tin hiện tại sẽ được giữ để bạn có thể in hồ sơ cho đến khi tạo phiên mới.*"
        )
        next_action = "session_complete"
    else:
        assist_msg = parsed.assistant_message
        next_action = "ask_for_missing_slots"

    return ChatOut(
        assistant_message=assist_msg,
        filled_slots=filled,
        missing_slots=missing,
        next_action=next_action,
        debug={
            "llm_next_action": parsed.next_action,
            "off_topic_count": get_off_topic_count(user_id),
            "is_off_topic": parsed.is_off_topic
        }
    )

@app.get("/patients")
def get_all_patients(db: Session = Depends(get_db)):
    """Get all patient records from database"""
    rows = db.query(PatientRecord).order_by(PatientRecord.created_at.desc()).all()
    return [
        {
            "record_id": r.record_id,
            "user_id": r.user_id,
            # Patient Information
            "patient_name": r.patient_name,
            "patient_phone": r.patient_phone,
            "patient_age": r.patient_age,
            "patient_gender": r.patient_gender,
            # Medical Information
            "symptoms": r.symptoms,
            "onset": r.onset,
            "allergies": r.allergies,
            "current_medications": r.current_medications,
            "pain_scale": r.pain_scale,
            # Results
            "predicted_diseases": json.loads(r.predicted_diseases) if r.predicted_diseases else {},
            "recommended_department": r.recommended_department,
            "created_at": r.created_at
        }
        for r in rows
    ]

@app.get("/statistics")
def get_statistics(db: Session = Depends(get_db)):
    """Get department statistics from patient records"""
    rows = db.query(PatientRecord.recommended_department).all()
    stats = {}
    for (dept,) in rows:
        dept = dept or "Unknown"
        stats[dept] = stats.get(dept, 0) + 1
    return stats

@app.get("/export_excel")
def export_excel(db: Session = Depends(get_db)):
    """Export patient records to Excel file"""
    try:
        rows = db.query(PatientRecord).order_by(PatientRecord.created_at.desc()).all()
        data = []
        
        for r in rows:
            data.append({
                "Record ID": r.record_id,
                "User ID": r.user_id,
                # Patient Information
                "Patient Name": r.patient_name or "",
                "Patient Phone": r.patient_phone or "",
                "Patient Age": r.patient_age or "",
                "Patient Gender": r.patient_gender or "",
                # Medical Information
                "Symptoms": r.symptoms or "",
                "Onset": r.onset or "",
                "Allergies": r.allergies or "",
                "Current Medications": r.current_medications or "",
                "Pain Scale": r.pain_scale or "",
                # Results
                "Predicted Diseases": r.predicted_diseases or "",
                "Recommended Department": r.recommended_department or "",
                "Created At": str(r.created_at) if r.created_at else ""
            })
        
        df = pd.DataFrame(data)
        
        # Try Excel first, fallback to CSV if openpyxl not available
        try:
            import openpyxl
            output_path = os.path.join(os.getcwd(), "patient_records.xlsx")
            df.to_excel(output_path, index=False, engine='openpyxl')
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = "patient_records.xlsx"
        except ImportError:
            # Fallback to CSV
            output_path = os.path.join(os.getcwd(), "patient_records.csv")
            df.to_csv(output_path, index=False, encoding='utf-8')
            media_type = "text/csv"
            filename = "patient_records.csv"
        
        return FileResponse(
            output_path, 
            filename=filename,
            media_type=media_type
        )
        
    except Exception as e:
        return {"error": f"Failed to export: {str(e)}"}

# ========= MAIN =========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)