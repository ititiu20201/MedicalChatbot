const API_BASE = "http://127.0.0.1:8000";

const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const exportBtn = document.getElementById("export-btn");

function addMessage(sender, text) {
  const div = document.createElement("div");
  div.textContent = `${sender}: ${text}`;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

sendBtn.addEventListener("click", async () => {
  const text = userInput.value.trim();
  if (!text) return;
  addMessage("Bạn", text);
  userInput.value = "";

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    if (!res.ok) throw new Error("API lỗi");
    const data = await res.json();

    let reply = "Kết quả dự đoán: ";
    reply += Object.entries(data.predicted_diseases)
                   .map(([d, p]) => `${d} (${p})`)
                   .join(", ");
    reply += `. Khoa gợi ý: ${data.department}`;

    addMessage("Bot", reply);
  } catch (err) {
    addMessage("Bot", "Xin lỗi, có lỗi xảy ra khi gọi API.");
  }
});

exportBtn.addEventListener("click", async () => {
  try {
    window.location.href = `${API_BASE}/export_excel`;
  } catch (err) {
    addMessage("Bot", "Xuất Excel thất bại.");
  }
});
