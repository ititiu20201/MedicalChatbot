// ====== Minimal frontend logic for Medical Chatbot ======
// Adjust BASE_URL if your backend is hosted elsewhere
const BASE_URL = ""; // e.g., "http://127.0.0.1:8000" or empty if same origin

const els = {
chat: document.getElementById("chat"),
form: document.getElementById("chat-form"),
input: document.getElementById("message"),
attach: document.getElementById("btn-attach"),
file: document.getElementById("file-input"),
newChat: document.getElementById("btn-new"),
history: document.getElementById("btn-history"),
applyProfile: document.getElementById("btn-apply-profile"),
symptomChips: document.getElementById("symptom-chips"),
deptList: document.getElementById("dept-list"),
ticket: document.getElementById("btn-ticket"),
admin: {
stSessions: document.getElementById("st-sessions"),
stWait: document.getElementById("st-wait"),
stAvg: document.getElementById("st-avg"),
search: document.getElementById("search"),
log: document.getElementById("log"),
exportBtn: document.getElementById("btn-export"),
},
};