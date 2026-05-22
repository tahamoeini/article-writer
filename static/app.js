const taskListEl = document.getElementById("task-list");
const taskLogsEl = document.getElementById("task-logs");
const tasksEmptyEl = document.getElementById("tasks-empty");
const researchOutputEl = document.getElementById("research-output");
const healthOutputEl = document.getElementById("health-output");
const chatHistoryEl = document.getElementById("chat-history");
const pdfFileListEl = document.getElementById("pdf-file-list");
const pdfListStatusEl = document.getElementById("pdf-list-status");

let selectedTaskId = null;
let taskPollTimer = null;
let chatMessages = [];

function currentSettings() {
  return {
    OLLAMA_BASE_URL: document.getElementById("ollama-base-url").value.trim(),
    OLLAMA_CHAT_MODEL: document.getElementById("chat-model").value.trim(),
    OLLAMA_EMBED_MODEL: document.getElementById("embed-model").value.trim(),
    QDRANT_HOST: document.getElementById("qdrant-host").value.trim(),
    QDRANT_PORT: document.getElementById("qdrant-port").value.trim(),
    QDRANT_COLLECTION: document.getElementById("collection-name").value.trim(),
    GROBID_URL: document.getElementById("grobid-url").value.trim(),
    CHUNK_SIZES: document.getElementById("chunk-sizes").value.trim(),
    VECTOR_TOP_K: document.getElementById("vector-top-k").value.trim(),
    BM25_TOP_K: document.getElementById("bm25-top-k").value.trim(),
    FUSED_TOP_K: document.getElementById("fused-top-k").value.trim(),
  };
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "Request failed.");
  }
  return data;
}

function setText(el, text) {
  el.textContent = text;
}

function fillModelSelect(selectId, models) {
  const select = document.getElementById(selectId);
  const currentValue = select.value;
  const options = models.length ? models : [currentValue];
  select.replaceChildren();
  options.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    select.appendChild(option);
  });
  if (options.includes(currentValue)) {
    select.value = currentValue;
  }
}

async function loadModels() {
  setText(healthOutputEl, "Loading models...");
  try {
    const url = `/v1/models?ollama_base_url=${encodeURIComponent(currentSettings().OLLAMA_BASE_URL)}`;
    const data = await fetchJson(url, { method: "GET" });
    fillModelSelect("chat-model", data.models);
    fillModelSelect("embed-model", data.models);
    setText(healthOutputEl, `Loaded ${data.models.length} model(s).`);
  } catch (error) {
    setText(healthOutputEl, error.message);
  }
}

async function loadPdfFiles() {
  setText(pdfListStatusEl, "Loading PDF files...");
  try {
    const data = await fetchJson("/v1/corpus/pdfs", { method: "GET" });
    pdfFileListEl.replaceChildren();
    data.files.forEach((filename) => {
      const option = document.createElement("option");
      option.value = filename;
      option.textContent = filename;
      pdfFileListEl.appendChild(option);
    });
    setText(
      pdfListStatusEl,
      data.files.length
        ? `Loaded ${data.files.length} PDF file(s). Leave unselected to ingest all.`
        : "No PDF files found in corpus/pdfs."
    );
  } catch (error) {
    setText(pdfListStatusEl, error.message);
  }
}

function selectedPdfFiles() {
  return Array.from(pdfFileListEl.selectedOptions).map((option) => option.value);
}

function ingestWorkerCount() {
  const rawValue = Number.parseInt(document.getElementById("ingest-workers").value, 10);
  if (Number.isNaN(rawValue)) {
    return 1;
  }
  return Math.min(Math.max(rawValue, 1), 32);
}

function renderTask(task) {
  const item = document.createElement("button");
  item.type = "button";
  item.className = `task-item ${task.id === selectedTaskId ? "active" : ""}`;
  const title = document.createElement("div");
  title.className = "task-title";

  const name = document.createElement("strong");
  name.textContent = task.name;

  const status = document.createElement("span");
  status.className = "status-pill";
  status.textContent = task.status;

  title.append(name, status);

  const startedAt = document.createElement("small");
  startedAt.textContent = task.started_at;

  const metadata = document.createElement("small");
  metadata.textContent = JSON.stringify(task.metadata);

  item.append(title, startedAt, metadata);
  item.addEventListener("click", () => {
    selectedTaskId = task.id;
    refreshTasks();
  });
  return item;
}

async function refreshTasks() {
  try {
    const data = await fetchJson("/v1/tasks", { method: "GET" });
    taskListEl.replaceChildren();
    tasksEmptyEl.style.display = data.tasks.length ? "none" : "block";
    data.tasks.forEach((task) => taskListEl.appendChild(renderTask(task)));

    if (!selectedTaskId && data.tasks.length) {
      selectedTaskId = data.tasks[0].id;
    }
    if (selectedTaskId) {
      const selected = await fetchJson(`/v1/tasks/${selectedTaskId}`, { method: "GET" });
      if (selected.error) {
        setText(taskLogsEl, `${selected.log_text}\n\nERROR: ${selected.error}`);
      } else {
        setText(taskLogsEl, selected.log_text || JSON.stringify(selected.result || "No logs yet.", null, 2));
      }
    }

    const hasRunning = data.tasks.some((task) => task.status === "running");
    if (hasRunning && !taskPollTimer) {
      taskPollTimer = window.setInterval(refreshTasks, 2500);
    } else if (!hasRunning && taskPollTimer) {
      window.clearInterval(taskPollTimer);
      taskPollTimer = null;
    }
  } catch (error) {
    setText(taskLogsEl, error.message);
  }
}

async function startTask(url, payload) {
  const task = await fetchJson(url, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  selectedTaskId = task.id;
  await refreshTasks();
}

async function checkHealth() {
  setText(healthOutputEl, "Checking...");
  try {
    const data = await fetchJson("/health", { method: "GET" });
    setText(healthOutputEl, JSON.stringify(data, null, 2));
  } catch (error) {
    setText(healthOutputEl, error.message);
  }
}

async function runResearchQuery() {
  setText(researchOutputEl, "Running query...");
  try {
    const data = await fetchJson("/v1/research/query", {
      method: "POST",
      body: JSON.stringify({
        prompt: document.getElementById("research-prompt").value.trim(),
        settings_overrides: currentSettings(),
      }),
    });
    setText(researchOutputEl, `${data.text}\n\nCitations:\n${JSON.stringify(data.citations, null, 2)}`);
  } catch (error) {
    setText(researchOutputEl, error.message);
  }
}

function renderChat() {
  chatHistoryEl.replaceChildren();
  if (!chatMessages.length) {
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";
    const role = document.createElement("small");
    role.textContent = "Assistant";
    const content = document.createElement("div");
    content.textContent = "Chat responses will appear here.";
    bubble.append(role, content);
    chatHistoryEl.appendChild(bubble);
    return;
  }

  chatMessages.forEach((message) => {
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";
    bubble.dataset.role = message.role;
    const role = document.createElement("small");
    role.textContent = message.role;
    const content = document.createElement("div");
    content.textContent = message.content;
    bubble.append(role, content);
    chatHistoryEl.appendChild(bubble);
  });
  chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
}

async function sendChatMessage() {
  const promptEl = document.getElementById("chat-prompt");
  const prompt = promptEl.value.trim();
  if (!prompt) {
    return;
  }
  chatMessages.push({ role: "user", content: prompt });
  renderChat();
  promptEl.value = "";

  try {
    const data = await fetchJson("/v1/chat", {
      method: "POST",
      body: JSON.stringify({
        messages: chatMessages,
        settings_overrides: currentSettings(),
      }),
    });
    chatMessages.push({ role: "assistant", content: data.message });
  } catch (error) {
    chatMessages.push({ role: "assistant", content: `Error: ${error.message}` });
  }
  renderChat();
}

document.getElementById("models-button").addEventListener("click", loadModels);
document.getElementById("health-button").addEventListener("click", checkHealth);
document.getElementById("refresh-tasks-button").addEventListener("click", refreshTasks);
document.getElementById("refresh-pdfs-button").addEventListener("click", loadPdfFiles);
document.getElementById("select-all-pdfs-button").addEventListener("click", () => {
  Array.from(pdfFileListEl.options).forEach((option) => {
    option.selected = true;
  });
});
document.getElementById("clear-pdfs-button").addEventListener("click", () => {
  Array.from(pdfFileListEl.options).forEach((option) => {
    option.selected = false;
  });
});
document.getElementById("ingest-button").addEventListener("click", () =>
  startTask("/v1/tasks/ingest", {
    force: document.getElementById("ingest-force").checked,
    selected_files: selectedPdfFiles(),
    max_workers: ingestWorkerCount(),
    settings_overrides: currentSettings(),
  })
);
document.getElementById("build-button").addEventListener("click", () =>
  startTask("/v1/tasks/build-index", {
    recreate: document.getElementById("build-recreate").checked,
    settings_overrides: currentSettings(),
  })
);
document.getElementById("synthesis-button").addEventListener("click", () =>
  startTask("/v1/tasks/synthesis", {
    query: document.getElementById("synthesis-query").value.trim(),
    verbose: document.getElementById("synthesis-verbose").checked,
    settings_overrides: currentSettings(),
  })
);
document.getElementById("research-button").addEventListener("click", runResearchQuery);
document.getElementById("chat-button").addEventListener("click", sendChatMessage);

renderChat();
loadPdfFiles();
refreshTasks();
