# app/gemini_service.py
# Gọi Gemini API thật, ép buộc trả JSON đúng schema

import os
import json
import time
from typing import Dict, Any
import requests
from .schemas import LLMOutput, REQUIRED_SLOTS

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY chưa được đặt trong .env")

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

SYSTEM_PROMPT = (
    "Bạn là trợ lý y tế. Nhiệm vụ: thu thập đủ các trường bắt buộc "
    f"{REQUIRED_SLOTS}. Luôn trả JSON ĐÚNG SCHEMA:\n"
    "{\n"
    '  "assistant_message": "string",\n'
    '  "slots_extracted": {\n'
    '    "symptoms": "", "onset": "", "age": "", "gender": "",\n'
    '    "allergies": "", "current_medications": "", "pain_scale": ""\n'
    "  },\n"
    '  "missing_slots": ["..."],\n'
    '  "next_action": "ask_for_missing_slots" hoặc "call_phobert"\n'
    "}\n"
    "- Khi THIẾU slot: hỏi đúng 1 câu ngắn gọn về 1 slot quan trọng nhất còn thiếu.\n"
    "- Khi ĐỦ slot: next_action=call_phobert, assistant_message='Mình đã đủ thông tin để phân tích tiếp.'\n"
    "- KHÔNG chẩn đoán hay gợi ý khoa. KHÔNG trả thêm văn bản ngoài JSON."
)

def _build_payload(user_message: str, state: Dict[str, str]) -> Dict[str, Any]:
    missing = [s for s in REQUIRED_SLOTS if not state.get(s)]
    context = {
        "current_state": {k: v for k, v in state.items() if v},
        "missing_slots": missing
    }
    user_text = (
        "Người dùng vừa nói:\n"
        f"{user_message}\n\n"
        "Bối cảnh (state hiện có, chỉ để tham khảo):\n"
        f"{json.dumps(context, ensure_ascii=False)}\n\n"
        "Hãy TRẢ VỀ DUY NHẤT 1 JSON đúng schema ở trên."
    )

    return {
        "systemInstruction": {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {"role": "user", "parts": [{"text": user_text}]}
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 256,
            "responseMimeType": "application/json"
        }
    }

def _post(payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise ValueError(f"Gemini response unexpected: {e} | raw={data}")
    try:
        return json.loads(text)
    except Exception:
        raise ValueError(f"Gemini không trả JSON hợp lệ: {text[:200]}...")

def call_gemini(user_message: str, state: Dict[str, str]) -> Dict:
    """
    Gọi Gemini và enforce JSON schema (LLMOutput).
    Retry tối đa 3 lần nếu sai format.
    """
    payload = _build_payload(user_message, state)
    last_err = None

    for _ in range(3):
        try:
            data = _post(payload)
            return LLMOutput(**data).dict()
        except Exception as e:
            last_err = e
            payload["contents"][0]["parts"][0]["text"] += (
                "\n\nNHẮC LẠI: chỉ trả về JSON hợp lệ đúng schema, "
                "không thêm văn bản ngoài JSON."
            )
            time.sleep(0.5)

    raise RuntimeError(f"Gọi Gemini thất bại sau 3 lần: {last_err}")
