from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Chatbot API",
    description="Vietnamese Medical Chatbot using PhoBERT for disease prediction",
    version="1.0.0"
)

class ChatInput(BaseModel):
    message: str

class ChatResponse(BaseModel):
    predicted_diseases: dict
    message: str

class PhoBERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs[0][:, 0]  # <s> token
        return torch.sigmoid(self.fc(self.dropout(pooled)))

try:
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    NUM_LABELS = 86
    model = PhoBERTClassifier(num_labels=NUM_LABELS)
    state = torch.load("phobert_medchat_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    
    with open("labels.json", "r", encoding="utf-8") as f:
        id2label = json.load(f)
    
    logger.info("Model and labels loaded successfully")
    
except Exception as e:
    logger.error(f"Error loading model or labels: {e}")
    raise e

def idx2name(i):
    if isinstance(id2label, list):
        return id2label[i]
    return id2label[str(i)]

@app.get("/")
def read_root():
    return {"message": "Medical Chatbot API is running", "status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
def chat(input: ChatInput):
    try:
        if not input.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        inputs = tokenizer(
            input.message, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        with torch.no_grad():
            probs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            ).squeeze().tolist()
        
        # Show top predictions with lower threshold
        all_predictions = {idx2name(i): round(p, 2) for i, p in enumerate(probs)}
        top_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)[:5]
        
        prediction = {
            disease: prob for disease, prob in top_predictions
            if prob > 0.1  # Lower threshold
        }
        
        if not prediction:
            response_message = "Không tìm thấy bệnh nào phù hợp với triệu chứng này. Vui lòng mô tả chi tiết hơn."
        else:
            top_diseases = sorted(prediction.items(), key=lambda x: x[1], reverse=True)[:3]
            diseases_str = ", ".join([f"{disease} ({prob})" for disease, prob in top_diseases])
            response_message = f"Các bệnh có thể: {diseases_str}. Hãy tham khảo ý kiến bác sĩ để chẩn đoán chính xác."
        
        return ChatResponse(
            predicted_diseases=prediction,
            message=response_message
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)