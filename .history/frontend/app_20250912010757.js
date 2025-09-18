// ====== Minimal frontend logic for Medical Chatbot ======
].filter(Boolean).join("; ");


if (!intro) return;
renderMessage("user", "(Cập nhật hồ sơ) " + intro);
saveMessage("user", "(Cập nhật hồ sơ) " + intro);
});
}


if (els.symptomChips) {
els.symptomChips.addEventListener("click", (e) => {
const txt = e.target.closest(".chip")?.textContent;
if (!txt) return;
els.input.value = (els.input.value + (els.input.value ? ", " : "" ) + txt).trim();
els.input.focus();
});
}


if (els.deptList) {
els.deptList.addEventListener("click", (e) => {
const dept = e.target.closest(".dept")?.textContent;
if (!dept) return;
renderMessage("bot", `Bạn đang quan tâm chuyên khoa: ${dept}. Tôi có thể hỗ trợ sắp xếp lộ trình khám.`);
});
}


if (els.ticket) {
els.ticket.addEventListener("click", () => {
const num = Math.floor(100 + Math.random() * 900);
renderMessage("bot", `Đã tạo số thứ tự tạm thời: ${num}. Vui lòng tới quầy tiếp đón để xác nhận.`);
});
}


// ====== Admin page helpers (uses localStorage demo data) ======
if (els.admin.search) {
// Populate counters from local history
const keys = Object.keys(localStorage).filter((k) => k.startsWith("mc_history_"));
const all = keys.flatMap((k) => JSON.parse(localStorage.getItem(k) || "[]"));
const sessions = keys.length;
const tickets = all.filter((m) => /số thứ tự|lấy số/i.test(m.text || "")).length;
els.admin.stSessions.textContent = sessions;
els.admin.stWait.textContent = tickets;
const serverLatencyDemo = all.length ? Math.round(500 + Math.random() * 300) + " ms" : "–";
els.admin.stAvg.textContent = serverLatencyDemo;


const renderLog = (filter = "") => {
els.admin.log.innerHTML = "";
const items = all
.filter((m) => (m.text || "").toLowerCase().includes(filter.toLowerCase()))
.slice(-200);
for (const m of items) {
const div = document.createElement("div");
div.className = "log__item";
div.innerHTML = `<div><strong>${m.role === "user" ? "Người dùng" : "Bot"}</strong> • ${new Date(m.t).toLocaleString()}</div>
<div>${escapeHtml(m.text || "")}</div>`;
els.admin.log.appendChild(div);
}
};


renderLog();
els.admin.search.addEventListener("input", (e) => renderLog(e.target.value));
els.admin.exportBtn.addEventListener("click", () => {
const rows = ["timestamp,role,text"].concat(
all.map((m) => [m.t, m.role, JSON.stringify(m.text || "").replaceAll("\n", " ")].join(","))
);
const blob = new Blob([rows.join("\n")], { type: "text/csv" });
const a = document.createElement("a");
a.href = URL.createObjectURL(blob);
a.download = `chatlog_${new Date().toISOString().slice(0,10)}.csv`;
a.click();
});
}