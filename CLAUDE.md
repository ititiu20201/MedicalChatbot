# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Vietnamese Medical Chatbot System** - A comprehensive hospital-grade AI system combining conversational AI with Vietnamese medical expertise. The system uses **Gemini API** for natural language conversation management and **PhoBERT** for Vietnamese medical symptom classification, with a modern hospital-appropriate web interface.

### System Evolution
- **Phase 1-2**: Data preparation and PhoBERT fine-tuning (Jupyter notebooks)
- **Phase 3-4**: REST API development and database integration (Updated notebooks with production code)  
- **Phase 5**: Advanced conversation flow with Gemini API integration
- **Phase 6**: Production deployment with hospital-grade UI/UX and admin dashboard

### Current Production Features
- 🤖 **Conversational AI**: Natural Vietnamese dialogue using Gemini API with structured slot filling
- 📋 **Patient Information Collection**: Systematic collection of 9 required medical slots (name, phone, symptoms, etc.)
- 🏥 **Medical Classification**: PhoBERT-based disease prediction from Vietnamese symptoms
- 🎯 **Department Routing**: Intelligent mapping of predicted diseases to hospital departments  
- 💾 **Database Management**: Structured patient records with separate columns for analytics
- 🌐 **Modern Web Interface**: Hospital-appropriate UI with accessibility features
- 📊 **Admin Dashboard**: Comprehensive management system with statistics and export capabilities

## Key Commands

### Production System Startup
```bash
# 1. Start the FastAPI backend server
uvicorn app_chatbot:app --reload --port 8000
# Or directly: python app_chatbot.py

# 2. Serve the frontend (requires HTTP server)
# Option A: Using Python's built-in server
cd frontend && python -m http.server 5500

# Option B: Using Live Server (VSCode extension)
# Right-click on frontend/index.html → "Open with Live Server"

# 3. Access admin dashboard
# Open admin.html directly in browser or serve via HTTP server
```

### API Testing
```bash
# Test conversational chat endpoint (production)
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "message": "Xin chào, tôi bị ho và sốt"}'

# Test legacy prediction endpoint  
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Tôi bị ho và sốt suốt ba ngày nay"}'

# Health check
curl -X GET "http://127.0.0.1:8000/health"

# Get all patient records
curl -X GET "http://127.0.0.1:8000/patients"

# Get department statistics
curl -X GET "http://127.0.0.1:8000/statistics"
```

### Environment Setup
```bash
# Required environment variables in .env file:
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
DATABASE_URL=mysql+pymysql://root:123456@localhost:3306/medical_chatbot
MODEL_PATH=app/models/phobert_medchat_model.pt
ID2LABEL_PATH=app/assets/id2label.json

# Install dependencies
pip install fastapi uvicorn sqlalchemy pymysql python-dotenv
pip install torch transformers pandas openpyxl requests
```

### Development Notebooks
```bash
# Execute notebooks in sequence:
jupyter notebook module/01_eda_and_label_preparation.ipynb  # Data analysis & label prep
jupyter notebook module/02_finetune_phobert.ipynb          # PhoBERT fine-tuning  
jupyter notebook module/03_rest_api_chatbot.ipynb          # Production API development
jupyter notebook module/04_mysql_storage.ipynb             # Database integration
jupyter notebook module/05_dashboard_and_export.ipynb      # Analytics & export
jupyter notebook module/06_final_deployment_and_review.ipynb # Deployment guide

# Note: Notebooks 3 & 4 have been updated with production code
```

## Architecture

### Production Architecture

**Three-Tier Architecture:**
1. **Frontend Layer**: Modern hospital-appropriate web interfaces
   - `frontend/index.html` - Patient-facing chatbot interface
   - `admin.html` - Administrative dashboard for hospital staff
   - `frontend/config.js` - Environment configuration for deployment flexibility

2. **API Layer**: FastAPI service with comprehensive medical workflow
   - **Conversation Management**: Gemini API integration for natural Vietnamese dialogue
   - **Slot Filling System**: Systematic collection of 9 required medical fields
   - **Disease Classification**: PhoBERT model for Vietnamese symptom analysis
   - **Department Routing**: Intelligent mapping to appropriate medical specialties

3. **Data Layer**: Structured MySQL database with hospital-grade organization
   - **Patient Information**: Separate columns for demographics (name, phone, age, gender)
   - **Medical Records**: Clinical data (symptoms, onset, allergies, medications, pain scale)
   - **AI Results**: Disease predictions and department recommendations
   - **Analytics Support**: Indexed columns for hospital management queries

### Critical Production Files

**Backend Core:**
- `app_chatbot.py` - Main FastAPI application with complete medical workflow
- `app/gemini_service.py` - Gemini API integration for conversational AI
- `app/schemas.py` - Pydantic models and validation schemas
- `app/state_store.py` - In-memory conversation state management
- `app/models.py` - SQLAlchemy database models with separate columns
- `app/db.py` - Database connection and session management

**Model Assets:**
- `app/models/phobert_medchat_model.pt` - Fine-tuned PhoBERT weights (86 disease labels)
- `app/assets/id2label.json` - Disease ID to label mappings
- `app/department_map.json` - Disease to department routing rules

**Frontend Interface:**
- `frontend/index.html` - Modern hospital-appropriate patient interface
- `frontend/app.js` - Chat functionality with Vietnamese IME support
- `frontend/config.js` - Deployment configuration for API endpoints
- `admin.html` - Comprehensive admin dashboard for hospital staff

**Database Management:**
- `migrate_db.py` - Database migration scripts for schema updates
- `.env` - Environment configuration (API keys, database URLs)

**Development Assets:**
- `module/03_rest_api_chatbot.ipynb` - Updated with production API code
- `module/04_mysql_storage.ipynb` - Updated with production database integration

### Model Architecture
```
PhoBERTClassifier:
├── PhoBERT base (vinai/phobert-base)
├── Dropout(0.3) 
└── Linear(768 → 86 labels)
```

### Production Data Flow

**Complete Medical Consultation Workflow:**

1. **Patient Input** → Vietnamese symptom description via web interface
2. **Conversation Management** → Gemini API processes natural language
3. **Slot Filling** → Systematic extraction of 9 required medical fields:
   - Patient Info: name, phone_number, age, gender  
   - Clinical Data: symptoms, onset, allergies, current_medications, pain_scale
4. **Validation & State Management** → Track conversation progress (X/9 completed)
5. **Disease Classification** → PhoBERT processes complete symptom description
6. **Department Mapping** → Intelligent routing using regex rules and lookup tables
7. **Database Storage** → Structured record with separate columns for analytics
8. **Response Generation** → Medical recommendations with department guidance

### Critical Dependencies
- Model expects exactly 86 labels (NUM_LABELS = 86)
- Label mappings must match training data order
- Vietnamese text preprocessing through PhoBERT tokenizer
- Model loading requires `labels.json`, `id2label.json`, `label2id.json`

### Production API Endpoints

**Patient Interface:**
- `POST /chat` - Main conversational endpoint with full medical workflow
  - Input: `{"user_id": "string", "message": "Vietnamese text"}`
  - Output: Assistant response, filled slots, missing slots, next action
  - Features: Gemini conversation, slot filling, PhoBERT classification, database storage

**Legacy/Direct Access:**
- `POST /predict` - Direct disease prediction (bypasses conversation)
  - Input: `{"text": "Vietnamese symptoms"}`
  - Output: Disease predictions with confidence scores and department recommendation

**System Management:**
- `GET /health` - Comprehensive system status check
  - Reports: Model status, device info, database connectivity, slot configuration
- `GET /patients` - Retrieve all patient records with structured data
- `GET /statistics` - Hospital analytics and department utilization metrics
- `GET /export_excel` - Export patient data for hospital administration systems

### Frontend Interface Features

**Patient Chatbot (`frontend/index.html`):**
- 🏥 Hospital-appropriate modern design with medical branding
- 🗣️ Vietnamese Input Method Editor (IME) support with proper text clearing
- 📊 Real-time progress tracking (X/9 slots filled) with visual indicators  
- ♿ Full accessibility support (ARIA labels, keyboard navigation, high contrast)
- 📱 Responsive design for tablets and mobile devices used in hospitals
- 🔄 Session persistence with automatic reconnection capabilities

**Admin Dashboard (`admin.html`):**
- 👥 Patient record management with search and filtering capabilities
- 📈 Real-time statistics with interactive charts (Chart.js integration)
- 📋 Detailed patient information display with medical history
- 📊 Department utilization analytics for hospital resource planning
- 📤 Excel export functionality for integration with hospital systems
- 🔍 Advanced search by patient name, phone, or department

### Database Schema (Production)

**Organized Column Structure:**
```sql
patient_records:
├── id (Primary Key)
├── record_id (Unique identifier)
├── user_id (Session tracking)
├── Patient Information:
│   ├── patient_name (Indexed for search)
│   ├── patient_phone (Indexed for search) 
│   ├── patient_age
│   └── patient_gender
├── Medical Records:
│   ├── symptoms (Text)
│   ├── onset (When symptoms started)
│   ├── allergies (Drug/food allergies)
│   ├── current_medications (Current treatments)
│   └── pain_scale (1-10 pain level)
├── AI Results:
│   ├── predicted_diseases (JSON of PhoBERT output)
│   └── recommended_department (Indexed for analytics)
└── System Metadata:
    ├── chat (Full conversation context)
    └── created_at (Indexed for time-based queries)
```

### Deployment Configuration

**Frontend/Backend Separation:**
- `frontend/config.js` enables independent deployment of frontend and backend
- Configurable API base URLs for development vs production environments
- CORS configuration supports multiple frontend origins
- Environment-based feature flags for different deployment scenarios

**Vietnamese Phone Number Validation:**
- Supports Vietnamese format: 10 digits starting with 0 or +84
- Integrated into Gemini prompts for proper data collection
- Database field optimized for Vietnamese phone number storage

### Known Issues & Solutions

**Resolved Issues:**
- ✅ Vietnamese IME text clearing - Fixed with input field replacement technique
- ✅ Name and phone number collection - Enhanced Gemini prompts with hospital priorities  
- ✅ Database schema organization - Migrated to separate columns for better analytics
- ✅ Admin panel JSON parsing - Added fallback handling for mixed data formats
- ✅ Frontend/backend coupling - Implemented configurable deployment architecture

**Current Monitoring:**
- Multiple background processes may accumulate - monitor with `ps aux | grep python`
- Gemini API rate limits - implement exponential backoff if needed
- Database connection pooling - monitor MySQL connection limits in production
- PhoBERT model memory usage - monitor GPU/CPU memory allocation