// ====== Minimal frontend logic for Medical Chatbot ======
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


// ====== Event wiring ======
if (els.form) {
els.form.addEventListener("submit", async (e) => {
e.preventDefault();
const text = (els.input.value || "").trim();
if (!text) return;


renderMessage("user", text);
saveMessage("user", text);
els.input.value = "";


// Placeholder typing indicator
const typing = document.createElement("div");
typing.className = "msg msg--bot";
typing.innerHTML = `<div class="msg__avatar">🤖</div><div class="msg__bubble">Đang soạn…</div>`;
els.chat.appendChild(typing);
els.chat.scrollTop = els.chat.scrollHeight;


try {
const data = await sendToChatbot(text);
typing.remove();
const reply = data.reply || "(Không có phản hồi)";
renderMessage("bot", reply);
saveMessage("bot", reply);
} catch (err) {
typing.remove();
const msg = `Xin lỗi, có lỗi kết nối máy chủ. ${err?.message || ""}`;
renderMessage("bot", msg);
saveMessage("bot", msg);
}
}