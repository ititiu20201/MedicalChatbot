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
let sessionId = localStorage.getItem("mc_session") || createSession();
const id = "s_" + Math.random().toString(36).slice(2) + Date.now().toString(36);
localStorage.setItem("mc_session", id);
localStorage.setItem("mc_history_" + id, JSON.stringify([]));
return id;
}


function saveMessage(role, text) {
const key = "mc_history_" + sessionId;
const list = JSON.parse(localStorage.getItem(key) || "[]");
list.push({ t: Date.now(), role, text });
localStorage.setItem(key, JSON.stringify(list));
}


function renderMessage(role, text) {
const wrap = document.createElement("div");
wrap.className = `msg ${role === "user" ? "msg--user" : "msg--bot"}`;
wrap.innerHTML = `
<div class="msg__avatar">${role === "user" ? "🧑" : "🤖"}</div>
<div class="msg__bubble">${escapeHtml(text)}</div>
`;
els.chat.appendChild(wrap);
els.chat.scrollTop = els.chat.scrollHeight;
}


function escapeHtml(s) {
return s
.replaceAll("&", "&amp;")
.replaceAll("<", "&lt;")
.replaceAll(">", "&gt;")
.replaceAll('"', "&quot;")
.replaceAll("'", "&#39;");
}


async function sendToChatbot(text, payload = {}) {
const url = BASE_URL + "/chat"; // Expect FastAPI endpoint that returns { reply: "..." }
const body = { session_id: sessionId, message: text, ...payload };


const res = await fetch(url, {
method: "POST",
headers: { "Content-Type": "application/json" },
body: JSON.stringify(body),
});
if (!res.ok) throw new Error("HTTP " + res.status);
return await res.json();
}