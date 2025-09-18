# app/gemini_service.py
# Gọi Gemini API thật, ép buộc trả JSON đúng schema

import os
import json
import time
from typing import Dict, Any
import requests
from .schemas import LLMOutput, REQUIRED_SLOTS

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

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
    '    "name": "", "phone_number": "", "symptoms": "", "onset": "",\n'
    '    "age": "", "gender": "", "allergies": "", "current_medications": "", "pain_scale": ""\n'
    "  },\n"
    '  "missing_slots": ["..."],\n'
    '  "next_action": "ask_for_missing_slots" hoặc "call_phobert"\n'
    "}\n"
    "THỨ TỰ ƯU TIÊN TẠI PHÒNG KHÁM:\n"
    "1. name (họ và tên đầy đủ) - BẮT BUỘC THU THẬP TRƯỚC\n"
    "2. phone_number (số điện thoại liên lạc 10-11 số) - BẮT BUỘC THU THẬP TRƯỚC\n"  
    "3. symptoms (triệu chứng chính)\n"
    "4. các thông tin khác: onset, age, gender, allergies, current_medications, pain_scale\n"
    "HƯỚNG DẪN CHI TIẾT:\n"
    "- name: HỎI TRƯỚC TIÊN - họ và tên đầy đủ của bệnh nhân\n"
    "- phone_number: HỎI SAU TÊN - số điện thoại Việt Nam (10 số bắt đầu bằng 0 hoặc +84)\n"
    "- symptoms: triệu chứng, biểu hiện bệnh lý\n"
    "- onset: thời gian bắt đầu (ví dụ: 3 ngày trước, 1 tuần)\n" 
    "- age: tuổi của bệnh nhân (chỉ số)\n"
    "- gender: giới tính (nam/nữ)\n"
    "- allergies: dị ứng thuốc/thức ăn. Nếu bệnh nhân trả lời 'không có', 'không', 'chưa từng' thì ghi 'không có dị ứng'\n"
    "- current_medications: thuốc đang uống hiện tại. Nếu 'không có' thì ghi 'không uống thuốc'\n"
    "- pain_scale: mức độ đau từ 1-10. CHẤP NHẬN chỉ số (1,2,3...10) hoặc mô tả (nhẹ=1-3, vừa=4-6, nặng=7-10)\n"
    "QUY TẮC:\n"
    "- LUÔN LUÔN bao gồm 'name' và 'phone_number' trong missing_slots nếu chưa có\n"
    "- Khi THIẾU slot: hỏi đúng 1 câu ngắn gọn về 1 slot quan trọng nhất còn thiếu (ưu tiên name trước)\n"
    "- Khi ĐỦ slot: next_action=call_phobert, assistant_message='Mình đã đủ thông tin để phân tích tiếp.'\n"
    "- LUÔN XÁC NHẬN thông tin dị ứng và thuốc đang dùng một cách rõ ràng.\n"
    "- CHẤP NHẬN pain_scale là số đơn giản như '5', '8', không cần câu đầy đủ.\n"
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
