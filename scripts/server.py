import contextlib
import io
import os
import sys
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, model_validator

try:
    from scripts.config import RuntimeSettings
except ModuleNotFoundError:
    from config import RuntimeSettings


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
OLLAMA_LIST_TIMEOUT_SECONDS = 30.0
OLLAMA_CHAT_TIMEOUT_SECONDS = 300.0

app = FastAPI(title="Article Writer Control Center")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class ResearchQuery(BaseModel):
    prompt: str = Field(min_length=3, description="Research question or prompt.")
    settings_overrides: dict[str, str] | None = None


class IngestRequest(BaseModel):
    force: bool = False
    settings_overrides: dict[str, str] | None = None


class BuildIndexRequest(BaseModel):
    recreate: bool = False
    settings_overrides: dict[str, str] | None = None


class SynthesisRequest(BaseModel):
    query: str = Field(min_length=3)
    verbose: bool = False
    settings_overrides: dict[str, str] | None = None


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    prompt: str | None = Field(default=None, min_length=1)
    messages: list[ChatMessage] = Field(default_factory=list)
    settings_overrides: dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_prompt_or_messages(self):
        if not self.prompt and not self.messages:
            raise ValueError("Either prompt or messages must be provided.")
        return self


@dataclass
class TaskRecord:
    id: str
    name: str
    status: str
    started_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    ended_at: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["log_text"] = "".join(self.logs)
        return payload


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_overrides(settings_overrides: Mapping[str, Any] | None) -> dict[str, str]:
    return {
        str(key): str(value).strip()
        for key, value in (settings_overrides or {}).items()
        if value is not None and str(value).strip()
    }


def _cache_key(settings_overrides: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(_normalize_overrides(settings_overrides).items()))


@lru_cache(maxsize=8)
def _get_engine_cached(cache_key: tuple[tuple[str, str], ...]):
    overrides = dict(cache_key)
    try:
        from scripts.query_engine import get_advanced_query_engine
    except ModuleNotFoundError:
        from query_engine import get_advanced_query_engine

    return get_advanced_query_engine(settings_overrides=overrides or None)


def get_engine(settings_overrides: Mapping[str, Any] | None = None):
    return _get_engine_cached(_cache_key(settings_overrides))


def reset_engine_cache() -> None:
    _get_engine_cached.cache_clear()


def _extract_models(response: Any) -> list[str]:
    models = getattr(response, "models", None)
    if models is None and isinstance(response, Mapping):
        models = response.get("models")
    extracted: list[str] = []
    for entry in models or []:
        if isinstance(entry, Mapping):
            name = entry.get("model") or entry.get("name")
        else:
            name = getattr(entry, "model", None) or getattr(entry, "name", None)
        if name:
            extracted.append(str(name))
    return sorted(dict.fromkeys(extracted))


def _extract_chat_content(response: Any) -> str:
    message = getattr(response, "message", None)
    if message is None and isinstance(response, Mapping):
        message = response.get("message", {})
    if isinstance(message, Mapping):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _serialize_result(result: Any) -> Any:
    if isinstance(result, tuple):
        return list(result)
    if isinstance(result, (dict, list, str, int, float, bool)) or result is None:
        return result
    return str(result)


def _serialize_citations(source_nodes: list[Any]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for source_node in source_nodes:
        node = getattr(source_node, "node", None)
        metadata = getattr(node, "metadata", {}) or {}
        citations.append(
            {
                "title": metadata.get("title"),
                "author": metadata.get("author"),
                "year": metadata.get("year"),
                "page": metadata.get("page"),
                "paragraph": metadata.get("paragraph"),
                "score": getattr(source_node, "score", None),
            }
        )
    return citations


def _run_ingest(force: bool, settings_overrides: Mapping[str, str] | None = None):
    try:
        from scripts.ingest import process_corpus
    except ModuleNotFoundError:
        from ingest import process_corpus

    return process_corpus(force=force, settings_overrides=settings_overrides)


def _run_build_index(recreate: bool, settings_overrides: Mapping[str, str] | None = None):
    try:
        from scripts.build_index import build_hierarchical_index
    except ModuleNotFoundError:
        from build_index import build_hierarchical_index

    result = build_hierarchical_index(recreate=recreate, settings_overrides=settings_overrides)
    reset_engine_cache()
    return result


def _run_synthesis(query: str, verbose: bool, settings_overrides: Mapping[str, str] | None = None):
    try:
        from scripts.synthesis import run_global_literature_review
    except ModuleNotFoundError:
        from synthesis import run_global_literature_review

    return run_global_literature_review(
        broad_query=query,
        verbose=verbose,
        settings_overrides=settings_overrides,
    )


class _TaskLogWriter(io.TextIOBase):
    def __init__(self, callback):
        self._callback = callback

    def write(self, data: str) -> int:
        if data:
            self._callback(data)
        return len(data)

    def flush(self) -> None:
        return None


class _ThreadLocalStream(io.TextIOBase):
    def __init__(self, fallback: io.TextIOBase):
        self._fallback = fallback
        self._local = threading.local()

    def _current_stream(self) -> io.TextIOBase:
        return getattr(self._local, "stream", self._fallback)

    @contextlib.contextmanager
    def redirect(self, stream: io.TextIOBase):
        previous = getattr(self._local, "stream", None)
        self._local.stream = stream
        try:
            yield
        finally:
            if previous is None:
                del self._local.stream
            else:
                self._local.stream = previous

    def write(self, data: str) -> int:
        return self._current_stream().write(data)

    def flush(self) -> None:
        self._current_stream().flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._current_stream(), name)


def _install_thread_local_stream(name: str) -> _ThreadLocalStream:
    stream = getattr(sys, name)
    if isinstance(stream, _ThreadLocalStream):
        return stream
    wrapped = _ThreadLocalStream(stream)
    setattr(sys, name, wrapped)
    return wrapped


TASK_STDOUT = _install_thread_local_stream("stdout")
TASK_STDERR = _install_thread_local_stream("stderr")


class TaskManager:
    MAX_LOG_ENTRIES = 2000

    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskRecord] = {}

    def create_task(self, name: str, metadata: dict[str, Any], target, **kwargs) -> TaskRecord:
        record = TaskRecord(
            id=str(uuid.uuid4()),
            name=name,
            status="running",
            started_at=utc_now(),
            metadata=metadata,
        )
        with self._lock:
            self._tasks[record.id] = record

        thread = threading.Thread(
            target=self._run_task,
            args=(record.id, target, kwargs),
            daemon=True,
        )
        thread.start()
        return record

    def _append_log(self, task_id: str, message: str) -> None:
        with self._lock:
            record = self._tasks[task_id]
            record.logs.append(message)
            if len(record.logs) > self.MAX_LOG_ENTRIES:
                record.logs = record.logs[-self.MAX_LOG_ENTRIES :]

    def _complete_task(self, task_id: str, *, result: Any = None, error: str | None = None) -> None:
        with self._lock:
            record = self._tasks[task_id]
            record.status = "failed" if error else "completed"
            record.result = _serialize_result(result)
            record.error = error
            record.ended_at = utc_now()

    def _run_task(self, task_id: str, target, kwargs: dict[str, Any]) -> None:
        writer = _TaskLogWriter(lambda message: self._append_log(task_id, message))
        try:
            with TASK_STDOUT.redirect(writer), TASK_STDERR.redirect(writer):
                result = target(**kwargs)
        except Exception as exc:
            self._append_log(task_id, "\n" + traceback.format_exc())
            self._complete_task(task_id, error=str(exc))
            return

        self._complete_task(task_id, result=result)

    def list_tasks(self) -> list[dict[str, Any]]:
        with self._lock:
            tasks = [record.to_payload() for record in self._tasks.values()]
        return sorted(tasks, key=lambda item: item["started_at"], reverse=True)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._tasks.get(task_id)
            return record.to_payload() if record else None


task_manager = TaskManager()


def _settings_defaults() -> dict[str, Any]:
    settings = RuntimeSettings.from_env()
    return {
        "qdrant_host": settings.qdrant_host,
        "qdrant_port": settings.qdrant_port,
        "collection_name": settings.collection_name,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_embed_model": settings.ollama_embed_model,
        "ollama_chat_model": settings.ollama_chat_model,
        "grobid_base_url": settings.grobid_base_url,
        "chunk_sizes": ",".join(str(value) for value in settings.chunk_sizes),
        "vector_top_k": settings.vector_top_k,
        "bm25_top_k": settings.bm25_top_k,
        "fused_top_k": settings.fused_top_k,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"defaults": _settings_defaults()},
    )


@app.get("/health")
async def health_check():
    try:
        get_engine()
    except Exception as exc:
        return {
            "status": "degraded",
            "detail": str(exc),
        }

    return {"status": "ok"}


@app.get("/v1/settings/defaults")
async def get_defaults():
    return _settings_defaults()


@app.get("/v1/models")
async def list_models(ollama_base_url: str | None = Query(default=None)):
    overrides = {"OLLAMA_BASE_URL": ollama_base_url} if ollama_base_url else None
    settings = RuntimeSettings.from_env(_normalize_overrides(overrides))
    try:
        models = _extract_models(settings.create_ollama_client(timeout=OLLAMA_LIST_TIMEOUT_SECONDS).list())
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"models": models}


@app.get("/v1/tasks")
async def list_tasks():
    return {"tasks": task_manager.list_tasks()}


@app.get("/v1/tasks/{task_id}")
async def get_task(task_id: str):
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task


@app.post("/v1/tasks/ingest")
async def start_ingest(payload: IngestRequest):
    overrides = _normalize_overrides(payload.settings_overrides)
    task = task_manager.create_task(
        "ingest",
        {"force": payload.force, "settings_overrides": overrides},
        _run_ingest,
        force=payload.force,
        settings_overrides=overrides or None,
    )
    return task.to_payload()


@app.post("/v1/tasks/build-index")
async def start_build_index(payload: BuildIndexRequest):
    overrides = _normalize_overrides(payload.settings_overrides)
    task = task_manager.create_task(
        "build-index",
        {"recreate": payload.recreate, "settings_overrides": overrides},
        _run_build_index,
        recreate=payload.recreate,
        settings_overrides=overrides or None,
    )
    return task.to_payload()


@app.post("/v1/tasks/synthesis")
async def start_synthesis(payload: SynthesisRequest):
    overrides = _normalize_overrides(payload.settings_overrides)
    task = task_manager.create_task(
        "synthesis",
        {
            "query": payload.query,
            "verbose": payload.verbose,
            "settings_overrides": overrides,
        },
        _run_synthesis,
        query=payload.query,
        verbose=payload.verbose,
        settings_overrides=overrides or None,
    )
    return task.to_payload()


@app.post("/v1/research/query")
async def execute_query(payload: ResearchQuery):
    try:
        response = get_engine(payload.settings_overrides).query(payload.prompt)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "text": str(response),
        "citations": _serialize_citations(getattr(response, "source_nodes", [])),
    }


@app.post("/v1/chat")
async def chat_with_model(payload: ChatRequest):
    overrides = _normalize_overrides(payload.settings_overrides)
    settings = RuntimeSettings.from_env(overrides or None)
    messages = [message.model_dump() for message in payload.messages]
    if payload.prompt:
        messages.append({"role": "user", "content": payload.prompt})

    try:
        response = settings.create_ollama_client(timeout=OLLAMA_CHAT_TIMEOUT_SECONDS).chat(
            model=settings.ollama_chat_model,
            messages=messages,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "model": settings.ollama_chat_model,
        "message": _extract_chat_content(response),
        "messages": messages,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
