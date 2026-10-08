"""Agent tasks: one conversation about one project folder, stored so any model can pick it up."""

from __future__ import annotations

import json
import time

from .state import connect

ACTIVE = ("running",)


def create(root: str, title: str, auto_run: bool = False) -> int:
    now = time.time()
    with connect() as c:
        cur = c.execute(
            "INSERT INTO tasks (root, title, status, plan, notes, auto_run, created, updated)"
            " VALUES (?, ?, 'idle', '[]', '', ?, ?, ?)",
            (root, title[:200], int(auto_run), now, now),
        )
        return int(cur.lastrowid)


def get(task_id: int) -> dict | None:
    with connect() as c:
        row = c.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if not row:
        return None
    task = dict(row)
    task["plan"] = json.loads(task["plan"] or "[]")
    task["auto_run"] = bool(task["auto_run"])
    return task


def list_for(root: str, limit: int = 30) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT id, title, status, provider, model, updated FROM tasks WHERE root = ? ORDER BY updated DESC LIMIT ?",
            (root, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def update(task_id: int, **fields) -> None:
    if "plan" in fields:
        fields["plan"] = json.dumps(fields["plan"])
    if "auto_run" in fields:
        fields["auto_run"] = int(fields["auto_run"])
    fields["updated"] = time.time()
    cols = ", ".join(f"{k} = ?" for k in fields)
    with connect() as c:
        c.execute(f"UPDATE tasks SET {cols} WHERE id = ?", (*fields.values(), task_id))


def mark_interrupted() -> None:
    """Tasks left 'running' by a process that died are not running any more."""
    with connect() as c:
        c.execute("UPDATE tasks SET status = 'stopped', error = 'Interrupted' WHERE status = 'running'")


# Messages. Roles: user, assistant, tool (sent to models) and note (shown to the user only).

def add_message(task_id: int, role: str, data: dict) -> int:
    now = time.time()
    with connect() as c:
        cur = c.execute(
            "INSERT INTO task_messages (task_id, role, data, ts) VALUES (?, ?, ?, ?)",
            (task_id, role, json.dumps(data), now),
        )
        c.execute("UPDATE tasks SET updated = ? WHERE id = ?", (now, task_id))
        return int(cur.lastrowid)


def messages(task_id: int, after: int = 0) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT id, role, data, ts FROM task_messages WHERE task_id = ? AND id > ? ORDER BY id",
            (task_id, after),
        ).fetchall()
    return [{"id": r["id"], "role": r["role"], "ts": r["ts"], **json.loads(r["data"])} for r in rows]


# Original file contents, saved the first time a task touches a file, so the task can be undone.

def remember_file(task_id: int, path: str, existed: bool, original: bytes | None) -> None:
    with connect() as c:
        c.execute(
            "INSERT OR IGNORE INTO task_files (task_id, path, existed, original, ts) VALUES (?, ?, ?, ?, ?)",
            (task_id, path, int(existed), original, time.time()),
        )


def changed_files(task_id: int) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT path, existed, original FROM task_files WHERE task_id = ? ORDER BY ts", (task_id,)
        ).fetchall()
    return [{"path": r["path"], "existed": bool(r["existed"]), "original": r["original"]} for r in rows]


def changed_paths(task_id: int) -> list[str]:
    with connect() as c:
        rows = c.execute("SELECT path FROM task_files WHERE task_id = ? ORDER BY ts", (task_id,)).fetchall()
    return [r["path"] for r in rows]


def forget_files(task_id: int) -> None:
    with connect() as c:
        c.execute("DELETE FROM task_files WHERE task_id = ?", (task_id,))
