const taskLogsEl = document.getElementById("task-logs");
const taskStatusEl = document.getElementById("task-status");
const researchOutputEl = document.getElementById("research-output");
const healthOutputEl = document.getElementById("health-output");
const chatHistoryEl = document.getElementById("chat-history");
const pdfFileListEl = document.getElementById("pdf-file-list");
const pdfListStatusEl = document.getElementById("pdf-list-status");

let selectedTaskId = null;
let taskPollTimer = null;
let chatMessages = [];
let lastLogText = "";

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
  const uniqueModels = Array.from(new Set(models.filter(Boolean)));
  const options = uniqueModels.includes(currentValue)
    ? uniqueModels
    : [currentValue, ...uniqueModels].filter(Boolean);
  select.replaceChildren();

  if (!options.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No Ollama models found";
    option.disabled = true;
    select.appendChild(option);
    return;
  }

  options.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model === currentValue && !uniqueModels.includes(model) ? `${model} (current)` : model;
    select.appendChild(option);
  });
  select.value = currentValue || options[0];
}

async function withButtonBusy(buttonId, busyText, action) {
  const button = document.getElementById(buttonId);
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = busyText;
  try {
    return await action();
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
}

async function loadModels() {
  setText(healthOutputEl, "Loading models...");
  return withButtonBusy("models-button", "Loading...", async () => {
    try {
      const url = `/v1/models?ollama_base_url=${encodeURIComponent(currentSettings().OLLAMA_BASE_URL)}`;
      const data = await fetchJson(url, { method: "GET" });
      fillModelSelect("chat-model", data.models);
      fillModelSelect("embed-model", data.models);
      setText(healthOutputEl, `Loaded ${data.models.length} model(s). Current selections were preserved.`);
    } catch (error) {
      setText(healthOutputEl, error.message);
    }
  });
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

async function refreshTasks() {
  try {
    const data = await fetchJson("/v1/tasks", { method: "GET" });
    const latestTask = data.tasks.find((task) => task.status === "running") || data.tasks[0];

    if (!latestTask) {
      selectedTaskId = null;
      lastLogText = "";
      setText(taskStatusEl, "No task has started yet.");
      setText(taskLogsEl, "Waiting for task output...");
      stopTaskPolling();
      return;
    }

    selectedTaskId = latestTask.id;
    setText(taskStatusEl, `${latestTask.name} · ${latestTask.status} · started ${latestTask.started_at}`);

    if (selectedTaskId) {
      const selected = await fetchJson(`/v1/tasks/${selectedTaskId}`, { method: "GET" });
      const logText = selected.log_text || "";
      const terminalText = logText || JSON.stringify(selected.result || "No output yet.", null, 2);

      if (selected.error) {
        setText(taskLogsEl, `${logText}\n\nERROR: ${selected.error}`);
      } else {
        setText(taskLogsEl, terminalText);
      }

      if (taskLogsEl.textContent !== lastLogText) {
        lastLogText = taskLogsEl.textContent;
        taskLogsEl.scrollTop = taskLogsEl.scrollHeight;
      }
    }

    const hasRunning = data.tasks.some((task) => task.status === "running");
    if (hasRunning) {
      startTaskPolling();
    } else {
      stopTaskPolling();
    }
  } catch (error) {
    setText(taskLogsEl, error.message);
  }
}

function startTaskPolling() {
  if (!taskPollTimer) {
    taskPollTimer = window.setInterval(refreshTasks, 1000);
  }
}

function stopTaskPolling() {
  if (taskPollTimer) {
    window.clearInterval(taskPollTimer);
    taskPollTimer = null;
  }
}

async function startTask(url, payload) {
  const task = await fetchJson(url, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  selectedTaskId = task.id;
  await refreshTasks();
  startTaskPolling();
}

async function checkHealth() {
  setText(healthOutputEl, "Checking...");
  return withButtonBusy("health-button", "Checking...", async () => {
    try {
      const data = await fetchJson("/health", { method: "GET" });
      setText(healthOutputEl, JSON.stringify(data, null, 2));
    } catch (error) {
      setText(healthOutputEl, error.message);
    }
  });
}

async function runResearchQuery() {
  const prompt = document.getElementById("research-prompt").value.trim();
  if (!prompt) {
    setText(researchOutputEl, "Enter a corpus question before asking.");
    return;
  }

  setText(researchOutputEl, "Running query...");
  return withButtonBusy("research-button", "Asking...", async () => {
    try {
      const data = await fetchJson("/v1/research/query", {
        method: "POST",
        body: JSON.stringify({
          prompt,
          settings_overrides: currentSettings(),
        }),
      });
      setText(researchOutputEl, `${data.text}\n\nCitations:\n${JSON.stringify(data.citations, null, 2)}`);
    } catch (error) {
      setText(researchOutputEl, error.message);
    }
  });
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
  withButtonBusy("ingest-button", "Starting...", () => {
    return startTask("/v1/tasks/ingest", {
      force: document.getElementById("ingest-force").checked,
      selected_files: selectedPdfFiles(),
      max_workers: ingestWorkerCount(),
      settings_overrides: currentSettings(),
    });
  })
);
document.getElementById("build-button").addEventListener("click", () =>
  withButtonBusy("build-button", "Starting...", () => {
    return startTask("/v1/tasks/build-index", {
      recreate: document.getElementById("build-recreate").checked,
      settings_overrides: currentSettings(),
    });
  })
);
document.getElementById("synthesis-button").addEventListener("click", () => {
  const query = document.getElementById("synthesis-query").value.trim();
  if (!query) {
    setText(taskLogsEl, "Enter a synthesis question before starting the task.");
    return;
  }

  return withButtonBusy("synthesis-button", "Starting...", () => {
    return startTask("/v1/tasks/synthesis", {
      query,
      verbose: document.getElementById("synthesis-verbose").checked,
      settings_overrides: currentSettings(),
    });
  });
});
document.getElementById("research-button").addEventListener("click", runResearchQuery);
document.getElementById("chat-button").addEventListener("click", () =>
  withButtonBusy("chat-button", "Sending...", sendChatMessage)
);

renderChat();
loadPdfFiles();
refreshTasks();
