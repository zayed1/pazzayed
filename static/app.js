const chat = document.getElementById("chat");
const form = document.getElementById("askForm");
const input = document.getElementById("question");
const statusEl = document.getElementById("status");
const uploadBtn = document.getElementById("uploadBtn");
const rebuildBtn = document.getElementById("rebuildBtn");
const fileInput = document.getElementById("fileInput");

function showEmpty() {
  chat.innerHTML = '<div class="empty">لا توجد رسائل بعد. اطرح سؤالاً للبدء.</div>';
}
showEmpty();

function addMessage(text, who) {
  const empty = chat.querySelector(".empty");
  if (empty) empty.remove();
  const el = document.createElement("div");
  el.className = `msg ${who}`;
  el.textContent = text;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

function renderSources(el, sources) {
  if (!sources || !sources.length) return;
  const details = document.createElement("details");
  details.className = "sources";
  const summary = document.createElement("summary");
  summary.textContent = `المصادر (${sources.length})`;
  details.appendChild(summary);
  for (const s of sources) {
    const item = document.createElement("div");
    item.className = "source-item";
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = `${s.source} — ${s.locator} (${s.score})`;
    item.appendChild(tag);
    const ex = document.createElement("div");
    ex.textContent = s.excerpt.length > 300 ? s.excerpt.slice(0, 300) + "…" : s.excerpt;
    item.appendChild(ex);
    details.appendChild(item);
  }
  el.appendChild(details);
}

async function refreshStatus() {
  try {
    const r = await fetch("/api/status");
    const s = await r.json();
    const ollama = s.ollama_available
      ? `<span class="ok">● النموذج المحلي متصل (${s.model})</span>`
      : `<span class="off">● النموذج المحلي غير متصل</span>`;
    const idx = s.index_built
      ? `<span class="ok">● الفهرس جاهز</span>`
      : `<span class="off">● الفهرس غير مبني</span>`;
    statusEl.innerHTML = `${ollama} ${idx} <span>الملفات: ${s.data_files.length}</span>`;
  } catch {
    statusEl.innerHTML = '<span class="off">تعذّر الاتصال بالخادم</span>';
  }
}
refreshStatus();

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  addMessage(question, "user");
  input.value = "";
  const loading = addMessage("جارٍ البحث في الكتب…", "bot");
  loading.classList.add("loading");

  try {
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await r.json();
    loading.remove();
    if (!r.ok) {
      addMessage(data.detail || "حدث خطأ.", "bot");
      return;
    }
    const el = addMessage(data.answer, "bot");
    renderSources(el, data.sources);
  } catch {
    loading.remove();
    addMessage("تعذّر الاتصال بالخادم.", "bot");
  }
});

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert("اختر ملف PDF أو CSV أولاً.");
    return;
  }
  uploadBtn.disabled = true;
  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await r.json();
    if (r.ok) {
      addMessage(`تم رفع الملف: ${data.saved}. اضغط "إعادة بناء الفهرس" لتضمينه.`, "bot");
      fileInput.value = "";
    } else {
      alert(data.detail || "فشل الرفع.");
    }
  } finally {
    uploadBtn.disabled = false;
    refreshStatus();
  }
});

rebuildBtn.addEventListener("click", async () => {
  rebuildBtn.disabled = true;
  rebuildBtn.textContent = "جارٍ البناء…";
  try {
    const r = await fetch("/api/rebuild", { method: "POST" });
    const data = await r.json();
    if (r.ok) {
      addMessage(`تم بناء الفهرس: ${data.chunks} مقطع (${data.backend}).`, "bot");
    } else {
      alert(data.detail || "فشل بناء الفهرس.");
    }
  } finally {
    rebuildBtn.disabled = false;
    rebuildBtn.textContent = "إعادة بناء الفهرس";
    refreshStatus();
  }
});
