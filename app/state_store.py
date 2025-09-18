# app/state_store.py
# Lưu state in-memory theo user_id; sau này có thể thay bằng Redis/DB.

from typing import Dict

_STATE: Dict[str, Dict[str, str]] = {}  # user_id -> slots dict

def get_state(user_id: str) -> Dict[str, str]:
    return _STATE.setdefault(user_id, {})

def merge_slots(user_id: str, new_slots: Dict[str, str]) -> Dict[str, str]:
    st = get_state(user_id)
    for k, v in (new_slots or {}).items():
        if v:  # không ghi đè giá trị tốt bằng chuỗi rỗng
            st[k] = v
    return st

def reset_state(user_id: str) -> None:
    _STATE.pop(user_id, None)
