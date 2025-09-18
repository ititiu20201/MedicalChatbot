// ====== Medical Chatbot Frontend (/app.js) ======
// Chỉnh sửa BASE_URL nếu backend chạy ở domain/port khác.
// VD: const BASE_URL = "http://127.0.0.1:8000";
const BASE_URL = ""; // để rỗng khi cùng origin

// --- Lấy phần tử DOM
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

// --- Quản lý phiên hội thoại (localStorage demo)
let sessionId = localStorage.getItem("mc_session") || createSession();

function createSession() {
  const id =
    "s_" + Math.random().toString(36).slice(2) + Date.now().toString(36);
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

function escapeHtml(s) {
  return (s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role === "user" ? "msg--user" : "msg--bot"}`;
  wrap.innerHTML = `
    <div class="msg__avatar">${role === "user" ? "🧑" : "🤖"}</div>
    <div class="msg__bubble">${escapeHtml(text)}</div>
  `;
  els.chat.appendChild(wapSafe(wrap));
  els.chat.scrollTop = els.chat.scrollHeight;
}

// Guard nhỏ chống null
function wapSafe(node) {
  return node || document.createTextNode("");
}

// --- Gửi text tới backend
async function sendToChatbot(text, payload = {}) {
  const url = BASE_URL + "/chat"; // FastAPI: trả về { reply: string }
  const body = { session_id: sessionId, message: text, ...payload };

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error("HTTP " + res.status);
  return await res.json();
}

// --- Auto-resize textarea & phím tắt gửi
function resizeTextarea() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(160, els.input.scrollHeight) + "px";
}
if (els.input) {
  ["input", "change"].forEach((ev) =>
    els.input.addEventListener(ev, resizeTextarea)
  );
  resizeTextarea();
  els.input.addEventListener("keydown", (e) => {
    // Enter để gửi, Shift+Enter để xuống dòng
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      els.form?.dispatchEvent(new Event("submit", { cancelable: true }));
    }
  });
}

// --- Submit chat
if (els.form) {
  els.form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = (els.input.value || "").trim();
    if (!text) return;

    renderMessage("user", text);
    saveMessage("user", text);
    els.input.value = "";
    resizeTextarea();

    // Gợi ý typing indicator
    const typing = document.createElement("div");
    typing.className = "msg msg--bot";
    typing.innerHTML = `
      <div class="msg__avatar">🤖</div>
      <div class="msg__bubble">Đang soạn…</div>
    `;
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
  });
}

// --- Đính kèm ảnh (UI stub; cần backend multipart để bật thật sự)
if (els.attach && els.file) {
  els.attach.addEventListener("click", () => els.file.click());
  els.file.addEventListener("change", () => {
    const f = els.file.files?.[0];
    if (!f) return;
    renderMessage(
      "user",
      `Đã chọn ảnh: ${f.name} (${Math.round(f.size / 1024)} KB).`
    );
    saveMessage(
      "user",
      `(Đính kèm ảnh) ${f.name} - ${Math.round(f.size / 1024)} KB`
    );
    // TODO: Gửi FormData tới endpoint upload của bạn khi sẵn sàng
  });
}

// --- Tạo phiên mới
if (els.newChat) {
  els.newChat.addEventListener("click", () => {
    sessionId = createSession();
    if (els.chat) els.chat.innerHTML = "";
    renderMessage(
      "bot",
      "Đã tạo phiên mới. Vui lòng mô tả vấn đề sức khỏe của bạn."
    );
  });
}

// --- Lịch sử nhanh (6 message gần nhất)
if (els.history) {
  els.history.addEventListener("click", () => {
    const key = "mc_history_" + sessionId;
    const list = JSON.parse(localStorage.getItem(key) || "[]");
    if (!list.length) {
      renderMessage("bot", "Chưa có lịch sử trong phiên này.");
      return;
    }
    const text =
      "Lịch sử gần đây:\n" +
      list
        .slice(-6)
        .map(
          (m) =>
            `${new Date(m.t).toLocaleTimeString()} • ${
              m.role === "user" ? "Bạn" : "Bot"
            }: ${m.text}`
        )
        .join("\n");
    renderMessage("bot", text);
  });
}

// --- Áp dụng hồ sơ bệnh nhân vào hội thoại
if (els.applyProfile) {
  els.applyProfile.addEventListener("click", () => {
    const f = document.getElementById("patient-form");
    if (!f) return;
    const data = Object.fromEntries(new FormData(f).entries());
    const intro = [
      data.name ? `Tên: ${data.name}` : null,
      data.age ? `Tuổi: ${data.age}` : null,
      data.gender ? `Giới tính: ${data.gender}` : null,
      data.phone ? `Liên hệ: ${data.phone}` : null,
      data.chief_complaint ? `Triệu chứng chính: ${data.chief_complaint}` : null,
    ]
      .filter(Boolean)
      .join("; ");

    if (!intro) return;
    const text = "(Cập nhật hồ sơ) " + intro;
    renderMessage("user", text);
    saveMessage("user", text);
  });
}

// --- Chips triệu chứng
if (els.symptomChips) {
  els.symptomChips.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    const txt = chip.textContent || "";
    els.input.value = (els.input.value + (els.input.value ? ", " : "") + txt)
      .trim();
    els.input.focus();
    resizeTextarea();
  });
}

// --- Gợi ý chuyên khoa
if (els.deptList) {
  els.deptList.addEventListener("click", (e) => {
    const dept = e.target.closest(".dept")?.textContent;
    if (!dept) return;
    renderMessage(
      "bot",
      `Bạn đang quan tâm chuyên khoa: ${dept}. Tôi có thể hỗ trợ sắp xếp lộ trình khám.`
    );
  });
}

// --- Lấy số thứ tự (demo)
if (els.ticket) {
  els.ticket.addEventListener("click", () => {
    const num = Math.floor(100 + Math.random() * 900);
    renderMessage(
      "bot",
      `Đã tạo số thứ tự tạm thời: ${num}. Vui lòng tới quầy tiếp đón để xác nhận.`
    );
  });
}

// --- Admin page (thống kê & export từ localStorage demo)
if (els.admin.search) {
  const keys = Object.keys(localStorage).filter((k) =>
    k.startsWith("mc_history_")
  );
  const all = keys.flatMap((k) =>
    JSON.parse(localStorage.getItem(k) || "[]")
  );
  const sessions = keys.length;
  const tickets = all.filter((m) =>
    /số thứ tự|lấy số/i.test(m.text || "")
  ).length;

  if (els.admin.stSessions) els.admin.stSessions.textContent = sessions;
  if (els.admin.stWait) els.admin.stWait.textContent = tickets;
  if (els.admin.stAvg)
    els.admin.stAvg.textContent = all.length
      ? Math.round(500 + Math.random() * 300) + " ms"
      : "–";

  const renderLog = (filter = "") => {
    if (!els.admin.log) return;
    els.admin.log.innerHTML = "";
    const items = all
      .filter((m) => (m.text || "").toLowerCase().includes(filter.toLowerCase()))
      .slice(-200);
    for (const m of items) {
      const div = document.createElement("div");
      div.className = "log__item";
      div.innerHTML = `<div><strong>${
        m.role === "user" ? "Người dùng" : "Bot"
      }</strong> • ${new Date(m.t).toLocaleString()}</div>
        <div>${escapeHtml(m.text || "")}</div>`;
      els.admin.log.appendChild(div);
    }
  };

  renderLog();
  els.admin.search.addEventListener("input", (e) => renderLog(e.target.value));
  els.admin.exportBtn?.addEventListener("click", () => {
    const rows = ["timestamp,role,text"].concat(
      all.map((m) =>
        [m.t, m.role, JSON.stringify(m.text || "").replaceAll("\n", " ")].join(
          ","
        )
      )
    );
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `chatlog_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
  });
}
