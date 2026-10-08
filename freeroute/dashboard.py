from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from . import state, tasks
from .agent import Agent
from .config import load_config
from .workspace import Terminal, Workspace, WorkspaceError

STATIC = Path(__file__).parent / "static"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    tasks.mark_interrupted()
    yield


app = FastAPI(title="freeroute dashboard", lifespan=lifespan)
terminal = Terminal()


@app.middleware("http")
async def local_only(request: Request, call_next):
    # The workspace API can run commands, so refuse other hostnames (DNS rebinding) and
    # cross-site writes: browsers will not send a custom header cross-origin without CORS approval.
    host = (request.headers.get("host") or "").rsplit(":", 1)[0]
    if host not in LOCAL_HOSTS:
        return PlainTextResponse("Only available on localhost", status_code=403)
    if request.method not in ("GET", "HEAD") and request.headers.get("x-freeroute") != "1":
        return PlainTextResponse("Missing X-Freeroute header", status_code=403)
    return await call_next(request)


@app.exception_handler(WorkspaceError)
async def workspace_error(_: Request, err: WorkspaceError) -> JSONResponse:
    return JSONResponse({"detail": str(err)}, status_code=400)


def _ws(root: str) -> Workspace:
    return Workspace(root)


def _task(task_id: int) -> dict:
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, f"No task {task_id}")
    return task


# Pages

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/workspace", response_class=HTMLResponse)
def workspace_page() -> str:
    return (STATIC / "workspace.html").read_text()


@app.get("/api/state")
def api_state() -> dict:
    # Re-read providers.yaml every poll so edits show up without a restart.
    return state.snapshot(load_config())


# Files

class OpenBody(BaseModel):
    root: str


class SaveBody(BaseModel):
    root: str
    path: str
    content: str


@app.post("/api/workspace")
def open_workspace(body: OpenBody) -> dict:
    ws = _ws(body.root)
    return {"root": str(ws.root), "name": ws.root.name, "tasks": tasks.list_for(str(ws.root))}


@app.get("/api/fs/list")
def fs_list(root: str, path: str = ".") -> list[dict]:
    return _ws(root).list_dir(path)


@app.get("/api/fs/file")
def fs_read(root: str, path: str) -> dict:
    ws = _ws(root)
    content = ws.read_text(path)
    return {"path": path, "content": content, "mtime": ws.resolve(path).stat().st_mtime}


@app.get("/api/fs/stat")
def fs_stat(root: str, path: str) -> dict:
    target = _ws(root).resolve(path)
    return {"exists": target.is_file(), "mtime": target.stat().st_mtime if target.is_file() else None}


@app.put("/api/fs/file")
def fs_write(body: SaveBody) -> dict:
    ws = _ws(body.root)
    ws.write_text(body.path, body.content)
    return {"mtime": ws.resolve(body.path).stat().st_mtime}


# Agent tasks

class Runs:
    """Agent runs happening in this process, plus commands waiting for the user's approval."""

    def __init__(self) -> None:
        self.threads: dict[int, threading.Thread] = {}
        self.stops: dict[int, threading.Event] = {}
        self.pending: dict[int, dict] = {}
        self.lock = threading.Lock()

    def running(self, task_id: int) -> bool:
        t = self.threads.get(task_id)
        return bool(t and t.is_alive())

    def start(self, task_id: int) -> None:
        with self.lock:
            if self.running(task_id):
                raise HTTPException(409, "The agent is already working on this task")
            stop = threading.Event()

            def approve(command: str) -> bool:
                waiting = {"command": command, "event": threading.Event(), "allow": False}
                self.pending[task_id] = waiting
                while not waiting["event"].wait(0.5):
                    if stop.is_set():
                        break
                self.pending.pop(task_id, None)
                return waiting["allow"]

            agent = Agent(task_id, approve=approve, stop=stop)
            thread = threading.Thread(target=agent.run, daemon=True, name=f"task-{task_id}")
            self.stops[task_id] = stop
            self.threads[task_id] = thread
            tasks.update(task_id, status="running", error=None)
            thread.start()

    def stop(self, task_id: int) -> None:
        if task_id in self.stops:
            self.stops[task_id].set()

    def answer(self, task_id: int, allow: bool) -> bool:
        waiting = self.pending.get(task_id)
        if not waiting:
            return False
        waiting["allow"] = allow
        waiting["event"].set()
        return True


runs = Runs()


class NewTask(BaseModel):
    root: str
    prompt: str
    auto_run: bool = False


class Message(BaseModel):
    prompt: str


class Approval(BaseModel):
    allow: bool
    always: bool = False


class Settings(BaseModel):
    auto_run: bool


@app.get("/api/tasks")
def list_tasks(root: str) -> list[dict]:
    return tasks.list_for(str(_ws(root).root))


@app.post("/api/tasks")
def create_task(body: NewTask) -> dict:
    ws = _ws(body.root)
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(400, "Describe the task first")
    task_id = tasks.create(str(ws.root), prompt.splitlines()[0], body.auto_run)
    tasks.add_message(task_id, "user", {"content": prompt})
    runs.start(task_id)
    return {"id": task_id}


@app.get("/api/tasks/{task_id}")
def get_task(task_id: int, after: int = 0) -> dict:
    task = _task(task_id)
    waiting = runs.pending.get(task_id)
    return {
        "task": task,
        "running": runs.running(task_id) or task["status"] == "running",
        "pending": {"command": waiting["command"]} if waiting else None,
        "changed": tasks.changed_paths(task_id),
        "messages": tasks.messages(task_id, after),
    }


@app.post("/api/tasks/{task_id}/messages")
def send_message(task_id: int, body: Message) -> dict:
    _task(task_id)
    if runs.running(task_id):
        raise HTTPException(409, "The agent is still working. Stop it first, or wait.")
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(400, "Empty message")
    tasks.add_message(task_id, "user", {"content": prompt})
    runs.start(task_id)
    return {"ok": True}


@app.post("/api/tasks/{task_id}/stop")
def stop_task(task_id: int) -> dict:
    _task(task_id)
    runs.stop(task_id)
    return {"ok": True}


@app.post("/api/tasks/{task_id}/approve")
def approve_command(task_id: int, body: Approval) -> dict:
    _task(task_id)
    if body.allow and body.always:
        tasks.update(task_id, auto_run=True)
    if not runs.answer(task_id, body.allow):
        raise HTTPException(409, "Nothing is waiting for approval")
    return {"ok": True}


@app.post("/api/tasks/{task_id}/settings")
def task_settings(task_id: int, body: Settings) -> dict:
    _task(task_id)
    tasks.update(task_id, auto_run=body.auto_run)
    return {"ok": True}


@app.post("/api/tasks/{task_id}/undo")
def undo_task(task_id: int) -> dict:
    task = _task(task_id)
    if runs.running(task_id):
        raise HTTPException(409, "Stop the agent before undoing its changes")
    restored = _ws(task["root"]).undo(task_id)
    if restored:
        tasks.add_message(task_id, "note", {"content": f"Undid changes to {len(restored)} file(s): "
                                                       + ", ".join(restored), "level": "warning"})
    return {"restored": restored}


# Terminal panel

class Command(BaseModel):
    root: str
    command: str


@app.post("/api/terminal")
def terminal_run(body: Command) -> dict:
    if not body.command.strip():
        raise HTTPException(400, "Empty command")
    return {"id": terminal.start(_ws(body.root), body.command)}


@app.get("/api/terminal/{job_id}")
def terminal_output(job_id: int, offset: int = 0) -> dict:
    job = terminal.get(job_id, offset)
    if not job:
        raise HTTPException(404, "No such command")
    return job


@app.post("/api/terminal/{job_id}/stop")
def terminal_stop(job_id: int) -> dict:
    return {"stopped": terminal.stop(job_id)}
