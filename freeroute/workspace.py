"""File and command access confined to one project folder, shared by the agent and the dashboard."""

from __future__ import annotations

import itertools
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from . import tasks

IGNORED = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".next", ".nuxt", "dist", "build", ".turbo", ".cache", ".idea", ".DS_Store", "target", ".gradle",
}
MAX_READ_BYTES = 2_000_000
OUTPUT_LIMIT = 12_000
SHELL = os.environ.get("SHELL") or "/bin/sh"


class WorkspaceError(ValueError):
    pass


class Workspace:
    def __init__(self, root: str | Path):
        path = Path(root).expanduser()
        if not path.is_dir():
            raise WorkspaceError(f"Not a folder: {root}")
        self.root = path.resolve()

    def resolve(self, rel: str) -> Path:
        rel = (rel or ".").strip()
        target = (self.root / rel).resolve()
        if target != self.root and not target.is_relative_to(self.root):
            raise WorkspaceError(f"{rel} is outside the project folder")
        return target

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."

    # Reading

    def list_dir(self, rel: str = ".") -> list[dict]:
        folder = self.resolve(rel)
        if not folder.is_dir():
            raise WorkspaceError(f"{rel} is not a folder")
        entries = []
        for child in folder.iterdir():
            if child.name in IGNORED:
                continue
            entries.append({"name": child.name, "path": self.rel(child), "dir": child.is_dir()})
        entries.sort(key=lambda e: (not e["dir"], e["name"].lower()))
        return entries

    def tree(self, rel: str = ".", depth: int = 2, limit: int = 300) -> str:
        lines: list[str] = []

        def walk(folder: Path, level: int) -> None:
            for entry in self.list_dir(self.rel(folder)):
                if len(lines) >= limit:
                    return
                lines.append("  " * level + entry["name"] + ("/" if entry["dir"] else ""))
                if entry["dir"] and level + 1 < depth:
                    walk(self.root / entry["path"], level + 1)

        walk(self.resolve(rel), 0)
        if len(lines) >= limit:
            lines.append(f"... (stopped at {limit} entries)")
        return "\n".join(lines) or "(empty folder)"

    def read_text(self, rel: str) -> str:
        path = self.resolve(rel)
        if not path.is_file():
            raise WorkspaceError(f"{rel} does not exist")
        if path.stat().st_size > MAX_READ_BYTES:
            raise WorkspaceError(f"{rel} is larger than 2 MB")
        data = path.read_bytes()
        if b"\0" in data[:8000]:
            raise WorkspaceError(f"{rel} is a binary file")
        return data.decode("utf-8", errors="replace")

    def search(self, pattern: str, rel: str = ".", limit: int = 200) -> str:
        target = self.resolve(rel)
        if shutil.which("rg"):
            cmd = ["rg", "--line-number", "--no-heading", "--color=never", "--max-columns=300"]
            cmd += [arg for name in IGNORED for arg in ("--glob", f"!{name}")]
            cmd += ["--", pattern, str(target)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode > 1:
                raise WorkspaceError(proc.stderr.strip() or "search failed")
            lines = [line.replace(str(self.root) + os.sep, "", 1) for line in proc.stdout.splitlines()]
        else:
            regex = re.compile(pattern)
            lines = []
            files = [target] if target.is_file() else (
                p for p in target.rglob("*") if p.is_file() and not IGNORED & set(p.relative_to(self.root).parts)
            )
            for path in files:
                try:
                    for n, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
                        if regex.search(line):
                            lines.append(f"{self.rel(path)}:{n}:{line[:300]}")
                except OSError:
                    continue
                if len(lines) > limit:
                    break
        if not lines:
            return "No matches."
        more = f"\n... {len(lines) - limit} more matches" if len(lines) > limit else ""
        return "\n".join(lines[:limit]) + more

    # Writing. Every change made for a task records the original first, so the task can be undone.

    def _remember(self, path: Path, task_id: int | None) -> None:
        if task_id is None:
            return
        existed = path.is_file()
        tasks.remember_file(task_id, self.rel(path), existed, path.read_bytes() if existed else None)

    def _writable(self, rel: str) -> Path:
        path = self.resolve(rel)
        if ".git" in path.relative_to(self.root).parts:
            raise WorkspaceError("Writing inside .git is not allowed")
        if path.is_dir():
            raise WorkspaceError(f"{rel} is a folder")
        return path

    def write_text(self, rel: str, content: str, task_id: int | None = None) -> bool:
        path = self._writable(rel)
        existed = path.exists()
        self._remember(path, task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return existed

    def edit_text(self, rel: str, old: str, new: str, replace_all: bool = False, task_id: int | None = None) -> int:
        text = self.read_text(rel)
        count = text.count(old) if old else 0
        if count == 0:
            raise WorkspaceError(f"old_text was not found in {rel}. Read the file again and copy the text exactly.")
        if count > 1 and not replace_all:
            raise WorkspaceError(
                f"old_text appears {count} times in {rel}. Include more surrounding lines, or set replace_all."
            )
        self.write_text(rel, text.replace(old, new) if replace_all else text.replace(old, new, 1), task_id)
        return count if replace_all else 1

    def delete(self, rel: str, task_id: int | None = None) -> None:
        path = self._writable(rel)
        if not path.is_file():
            raise WorkspaceError(f"{rel} does not exist")
        self._remember(path, task_id)
        path.unlink()

    def undo(self, task_id: int) -> list[str]:
        restored = []
        for f in tasks.changed_files(task_id):
            path = self.resolve(f["path"])
            if f["existed"]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f["original"] or b"")
            elif path.is_file():
                path.unlink()
            restored.append(f["path"])
        tasks.forget_files(task_id)
        return restored

    # Commands

    def run(self, command: str, timeout: float = 120) -> tuple[int | None, str]:
        """Run a command to completion. Returns (exit code or None on timeout, combined output)."""
        proc = subprocess.Popen(
            [SHELL, "-c", command], cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, errors="replace", start_new_session=True,
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
            code: int | None = proc.returncode
        except subprocess.TimeoutExpired:
            _kill(proc)
            out, _ = proc.communicate()
            code = None
            out += f"\n[stopped after {int(timeout)}s; long-running commands such as dev servers are not supported here]"
        return code, clip(out)


def clip(text: str, limit: int = OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return f"[... {len(text) - limit} characters cut ...]\n" + text[-limit:]


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=3)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


# Terminal panel: commands typed by the user run in the background and stream their output.

class Terminal:
    def __init__(self) -> None:
        self._ids = itertools.count(1)
        self._jobs: dict[int, dict] = {}
        self._lock = threading.Lock()

    def start(self, ws: Workspace, command: str) -> int:
        proc = subprocess.Popen(
            [SHELL, "-c", command], cwd=ws.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, errors="replace", bufsize=1, start_new_session=True,
        )
        job_id = next(self._ids)
        job = {"id": job_id, "command": command, "root": str(ws.root), "proc": proc, "output": "",
               "code": None, "done": False, "started": time.time()}
        with self._lock:
            self._jobs[job_id] = job
            for old in sorted(self._jobs)[:-50]:
                if self._jobs[old]["done"]:
                    del self._jobs[old]
        threading.Thread(target=self._pump, args=(job,), daemon=True).start()
        return job_id

    def _pump(self, job: dict) -> None:
        full = False
        for line in job["proc"].stdout:
            if full:
                continue
            job["output"] += line
            if len(job["output"]) > 1_000_000:
                job["output"] += "\n[output over 1 MB; the rest is not shown]\n"
                full = True
        job["code"] = job["proc"].wait()
        job["done"] = True

    def get(self, job_id: int, offset: int = 0) -> dict | None:
        job = self._jobs.get(job_id)
        if not job:
            return None
        out = job["output"]
        return {"id": job_id, "output": out[offset:], "offset": len(out), "done": job["done"], "code": job["code"]}

    def stop(self, job_id: int) -> bool:
        job = self._jobs.get(job_id)
        if not job or job["done"]:
            return False
        _kill(job["proc"])
        return True
