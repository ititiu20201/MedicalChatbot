# app/models.py
# ORM models cho bảng patient_records

from sqlalchemy import Column, String, Text
from .db import Base

class PatientRecord(Base):
    __tablename__ = "patient_records"

    record_id = Column(String(64), primary_key=True)
    user_id = Column(String(64), index=True)
    symptoms = Column(Text)                 # JSON string of filled slots
    predicted_diseases = Column(Text)       # JSON string of {label: prob}
    recommended_department = Column(String(128))
    chat = Column(Text)                     # optional: JSON string toàn bộ hội thoại/state
