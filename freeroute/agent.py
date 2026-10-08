"""The coding agent: works inside one project folder and keeps going when a model runs out of quota.

Models remember nothing between requests, so the task lives here, not in any model: the message
history, a plan the agent maintains, and the files on disk. Before each request the history is
fitted to the next model's context size, and switching model is just sending it somewhere else.
"""

from __future__ import annotations

import json
import platform
import re
import secrets
import string
import threading
from collections.abc import Callable
from datetime import datetime

from . import state, tasks
from .config import Model, Provider
from .router import NoModelAvailable, Result, Router
from .workspace import Workspace, WorkspaceError, clip

MAX_STEPS = 40
READ_LINES = 400
READ_CHARS = 40_000
CHARS_PER_TOKEN = 3.5
INPUT_SHARE = 0.75  # of a model's context; the rest is left for its reply
KEEP_FULL = 6  # most recent steps that are never shortened unless nothing else fits
GEMINI_DUMMY_SIGNATURE = "skip_thought_signature_validator"
THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.S)


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_PATH = {"type": "string", "description": "Path relative to the project folder"}
TOOLS = [
    _fn("list_files", "List files and folders.", {
        "path": {**_PATH, "description": "Folder relative to the project folder (default: the project folder)"},
        "depth": {"type": "integer", "description": "How many levels deep (default 2)"},
    }, []),
    _fn("read_file", f"Read a text file. Returns at most {READ_LINES} lines per call.", {
        "path": _PATH,
        "start_line": {"type": "integer", "description": "First line to read, 1-based (default 1)"},
        "end_line": {"type": "integer", "description": "Last line to read"},
    }, ["path"]),
    _fn("write_file", "Create a file or replace its whole content. Creates missing folders.", {
        "path": _PATH, "content": {"type": "string"},
    }, ["path", "content"]),
    _fn("edit_file", "Replace exact text in a file. old_text must match the file exactly and only once, "
        "unless replace_all is true.", {
        "path": _PATH, "old_text": {"type": "string"}, "new_text": {"type": "string"},
        "replace_all": {"type": "boolean"},
    }, ["path", "old_text", "new_text"]),
    _fn("delete_file", "Delete a file.", {"path": _PATH}, ["path"]),
    _fn("search", "Search file contents with a regular expression. Returns path:line:text matches.", {
        "pattern": {"type": "string"},
        "path": {**_PATH, "description": "File or folder to search (default: the project folder)"},
    }, ["pattern"]),
    _fn("run_command", "Run a shell command in the project folder and return its exit code and output. "
        "Commands stop after the timeout, so do not start dev servers or watchers.", {
        "command": {"type": "string"},
        "timeout": {"type": "integer", "description": "Seconds, default 120, max 600"},
    }, ["command"]),
    _fn("update_plan", "Save the task plan and notes. Another model may take over at any step and will rely on "
        "this, so keep it current: write it at the start, tick steps when done, note decisions.", {
        "steps": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "done": {"type": "boolean"}}, "required": ["text"]}},
        "notes": {"type": "string", "description": "Decisions, findings, and anything the next model should know"},
    }, ["steps"]),
]
TOOLS_TOKENS = int(len(json.dumps(TOOLS)) / CHARS_PER_TOKEN)

SYSTEM = """You are a coding agent working in the project folder {root} on {os}. Today is {today}.
Use the tools to inspect and change files and to run commands. All paths are relative to the project folder.

How to work:
- For anything beyond a one-line change, start with update_plan, then work through the steps.
- Read a file before editing it. Use edit_file for targeted changes and write_file for new files or full rewrites.
- Check your work when it makes sense: run the tests, the build, or the script you changed.
- After each finished step, call update_plan to tick it and record anything the next person needs to know.
  The model doing this task can change at any moment because of rate limits; the next one sees only the plan,
  the notes, the most recent steps and the files.
- When the task is done, reply with a short summary of what you changed, without calling a tool.
- If something is unclear or risky (deleting data, large rewrites), stop and ask instead of guessing."""


def _new_call_id() -> str:
    # Mistral only accepts 9-character alphanumeric tool-call ids; every other provider accepts them too.
    return "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(9))


def _head(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f" [... {len(text) - limit} more characters]"


def _tokens(messages: list[dict]) -> int:
    return int(len(json.dumps(messages, ensure_ascii=False)) / CHARS_PER_TOKEN) + TOOLS_TOKENS


def _arg(args: dict, key: str, default=None):
    value = args.get(key, default)
    return default if value is None else value


def describe_call(name: str, args: dict) -> str:
    path = args.get("path") or "."
    match name:
        case "list_files":
            return f"Listing {path}"
        case "read_file":
            return f"Reading {path}"
        case "write_file":
            return f"Writing {path}"
        case "edit_file":
            return f"Editing {path}"
        case "delete_file":
            return f"Deleting {path}"
        case "search":
            return f"Searching for {args.get('pattern', '')!r}"
        case "run_command":
            return f"Running `{_head(str(args.get('command', '')), 120)}`"
        case "update_plan":
            return "Updating the plan"
    return f"Calling {name}"


def call_args(call: dict) -> dict:
    try:
        args = json.loads(call.get("arguments") or "{}")
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}


class Agent:
    def __init__(
        self,
        task_id: int,
        *,
        approve: Callable[[str], bool] | None = None,
        on_message: Callable[[dict], None] | None = None,
        stop: threading.Event | None = None,
        router: Router | None = None,
    ):
        task = tasks.get(task_id)
        if not task:
            raise ValueError(f"No task {task_id}")
        self.task_id = task_id
        self.ws = Workspace(task["root"])
        self.approve = approve or (lambda command: False)
        self.on_message = on_message
        self.stop = stop or threading.Event()
        self.router = router or Router()
        self.session = f"task-{task_id}"
        self._too_long: set[tuple[str, str]] = set()
        self._tree = ""
        self._nudge = False

    # Storage

    def _add(self, role: str, data: dict) -> None:
        msg_id = tasks.add_message(self.task_id, role, data)
        if self.on_message:
            self.on_message({"id": msg_id, "role": role, **data})

    def note(self, text: str, level: str = "info") -> None:
        self._add("note", {"content": text, "level": level})

    # Main loop

    def run(self) -> str:
        history = tasks.messages(self.task_id)
        latest = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        tasks.update(self.task_id, status="running", error=None)
        request_id = state.start_request("agent", latest, session=self.session)
        status, error, empty_replies = "done", None, 0
        try:
            for _ in range(MAX_STEPS):
                if self.stop.is_set():
                    status = "stopped"
                    self.note("Stopped.", "warning")
                    break
                self._tree = self.ws.tree(depth=2, limit=120)
                self._too_long.clear()
                state.set_activity(request_id, "Choosing a model")
                result = self.router.chat(
                    request_id=request_id, session=self.session, prepare=self._prepare, tools=TOOLS,
                )
                self._note_switch(result)
                calls = self._save_reply(result)
                if calls:
                    empty_replies, self._nudge = 0, False
                    for call in calls:
                        self._execute(call, request_id)
                    continue
                if result.text.strip():
                    break
                empty_replies += 1
                if empty_replies > 2:
                    status, error = "failed", "The models kept returning empty replies."
                    self.note(error, "error")
                    break
                self._nudge = True
            else:
                status = "paused"
                self.note(f"Paused after {MAX_STEPS} steps. Send a message (for example \"continue\") to keep going.",
                          "warning")
        except NoModelAvailable as err:
            status, error = "failed", str(err)
            self.note(error, "error")
        except KeyboardInterrupt:
            status = "stopped"
            self.note("Stopped.", "warning")
        except Exception as err:  # noqa: BLE001 - surface any failure in the task instead of losing it
            status, error = "failed", f"{type(err).__name__}: {err}"
            self.note(error, "error")
        tasks.update(self.task_id, status=status, error=error)
        state.finish_request(request_id, ok=status != "failed", error=error)
        return status

    def _note_switch(self, result: Result) -> None:
        task = tasks.get(self.task_id)
        new = (result.provider.name, result.model.id)
        old = (task["provider"], task["model"]) if task["model"] else None
        if old == new:
            return
        tasks.update(self.task_id, provider=new[0], model=new[1])
        if old is None:
            self.note(f"Using {result.label}", "model")
            return
        status, until, last_error = state.model_state(*old)
        err = (last_error or "").lower()
        if old in self._too_long:
            why = "the conversation got too long for it"
        elif status == "exhausted":
            why = f"its quota is used up until {datetime.fromtimestamp(until).strftime('%-I:%M %p')}" if until \
                else "its quota is used up"
        elif "timed out" in err:
            why = "it timed out"
        elif status == "cooldown" and any(w in err for w in ("429", "rate", "quota", "limit")):
            why = "it hit a rate limit"
        elif status != "ready":
            why = f"it returned an error ({(last_error or 'unknown')[:120]})"
        else:
            why = "it did not answer"
        self.note(f"Switched from {old[0]}/{old[1]} to {result.label} because {why}. "
                  "It continues from the plan, the notes and the recent steps.", "model")

    def _save_reply(self, result: Result) -> list[dict]:
        msg = result.message
        calls = []
        for tc in msg.tool_calls or []:
            raw = tc.model_dump(exclude_none=True)
            fn = raw.get("function") or {}
            calls.append({
                "id": _new_call_id(),
                "name": fn.get("name") or "",
                "arguments": fn.get("arguments") or "{}",
                "extra": raw.get("extra_content"),
            })
        content = THINK_BLOCK.sub("", msg.content or "").strip()
        if content or calls:
            self._add("assistant", {
                "content": content, "tool_calls": calls,
                "provider": result.provider.name, "model": result.model.id,
            })
        return calls

    # Fitting the history to a model

    def _prepare(self, provider: Provider, model: Model) -> list[dict] | None:
        budget = int(provider.context_of(model) * INPUT_SHARE)
        history = [m for m in tasks.messages(self.task_id) if m["role"] != "note"]
        units = _group(history)
        if not units:
            return None
        target = (provider.name, model.id)
        gemini = "generativelanguage.googleapis.com" in provider.base_url
        first, rest = units[0], units[1:]
        dropped: list[list[dict]] = []

        def build(shorten_upto: int, shorten_first: bool = False) -> list[dict]:
            msgs = [{"role": "system", "content": self._system(dropped)}]
            for i, unit in enumerate([first, *rest]):
                short = (i == 0 and shorten_first) or (0 < i <= shorten_upto)
                msgs += _render(unit, target, gemini, short)
            if self._nudge:
                msgs.append({"role": "user", "content": "Continue the task: call a tool, or reply with a summary if "
                                                        "you are finished."})
            return msgs

        # Shorten older steps first, then drop the oldest ones, then shorten everything.
        for upto in (0, len(rest) - KEEP_FULL, len(rest) - 2):
            if upto < 0:
                continue
            msgs = build(upto)
            if _tokens(msgs) <= budget:
                return msgs
        while len(rest) > 1:
            idx = next((i for i, u in enumerate(rest[:-1]) if u[0]["role"] != "user"), 0)
            dropped.append(rest.pop(idx))
            msgs = build(len(rest) - 1)
            if _tokens(msgs) <= budget:
                return msgs
        msgs = build(len(rest), shorten_first=True)
        if _tokens(msgs) <= budget:
            return msgs
        self._too_long.add(target)
        return None

    def _system(self, dropped: list[list[dict]]) -> str:
        task = tasks.get(self.task_id)
        parts = [SYSTEM.format(root=self.ws.root, os=platform.system(), today=datetime.now().strftime("%Y-%m-%d"))]
        parts.append(f"Project files (two levels):\n{self._tree}")
        memory = []
        if task["plan"]:
            memory.append("Plan:\n" + "\n".join(f"[{'x' if s['done'] else ' '}] {s['text']}" for s in task["plan"]))
        if task["notes"]:
            memory.append(f"Notes:\n{task['notes']}")
        changed = tasks.changed_paths(self.task_id)
        if changed:
            memory.append("Files changed so far in this task: " + ", ".join(changed[:60]))
        if memory:
            parts.append("Task memory (kept across model switches):\n" + "\n\n".join(memory))
        if dropped:
            lines = [line for unit in dropped for line in _summarize(unit)]
            parts.append("Earlier steps, removed to fit the context window (re-read files if you need details):\n"
                         + "\n".join(lines[-60:]))
        return "\n\n".join(parts)

    # Tools

    def _execute(self, call: dict, request_id: int) -> None:
        name = call["name"]
        args = call_args(call)
        activity = describe_call(name, args)
        task = tasks.get(self.task_id)
        state.set_activity(request_id, activity, task["provider"], task["model"])
        state.add_event(request_id, activity)
        ok = False
        extra: dict = {}
        handler = getattr(self, f"_tool_{name}", None)
        if self.stop.is_set():
            result = "Not run: the user stopped the task."
        elif not handler:
            result = f"Error: there is no tool called {name!r}."
        elif not args and call.get("arguments", "{}").strip() not in ("", "{}"):
            result = "Error: the arguments were not valid JSON. Call the tool again with valid JSON arguments."
        else:
            try:
                result, extra = handler(args)
                ok = True
            except (WorkspaceError, KeyError, ValueError, TypeError) as err:
                missing = f"missing argument {err}" if isinstance(err, KeyError) else str(err)
                result = f"Error: {missing}"
            except OSError as err:
                result = f"Error: {err}"
        self._add("tool", {"tool_call_id": call["id"], "name": name, "content": result, "ok": ok, **extra})

    def _tool_list_files(self, args: dict) -> tuple[str, dict]:
        depth = max(1, min(int(_arg(args, "depth", 2)), 4))
        return self.ws.tree(str(_arg(args, "path", ".")), depth=depth), {}

    def _tool_read_file(self, args: dict) -> tuple[str, dict]:
        path = str(args["path"])
        lines = self.ws.read_text(path).splitlines()
        start = max(1, int(_arg(args, "start_line", 1)))
        end = min(len(lines), int(_arg(args, "end_line", start + READ_LINES - 1)), start + READ_LINES - 1)
        body = "\n".join(lines[start - 1:end])
        if len(body) > READ_CHARS:
            body = body[:READ_CHARS]
            end = start + body.count("\n")
        header = f"{path} (lines {start}-{end} of {len(lines)})" if (start > 1 or end < len(lines)) \
            else f"{path} ({len(lines)} lines)"
        more = f"\n[call read_file with start_line={end + 1} to read more]" if end < len(lines) else ""
        return f"{header}\n{body}{more}", {}

    def _tool_write_file(self, args: dict) -> tuple[str, dict]:
        path, content = str(args["path"]), str(args["content"])
        existed = self.ws.write_text(path, content, self.task_id)
        lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return f"{'Replaced' if existed else 'Created'} {path} ({lines} lines).", {"path": path}

    def _tool_edit_file(self, args: dict) -> tuple[str, dict]:
        path = str(args["path"])
        count = self.ws.edit_text(path, str(args["old_text"]), str(args["new_text"]),
                                  bool(_arg(args, "replace_all", False)), self.task_id)
        return f"Edited {path} ({count} replacement{'s' if count > 1 else ''}).", {"path": path}

    def _tool_delete_file(self, args: dict) -> tuple[str, dict]:
        path = str(args["path"])
        self.ws.delete(path, self.task_id)
        return f"Deleted {path}.", {"path": path}

    def _tool_search(self, args: dict) -> tuple[str, dict]:
        return self.ws.search(str(args["pattern"]), str(_arg(args, "path", "."))), {}

    def _tool_run_command(self, args: dict) -> tuple[str, dict]:
        command = str(args["command"]).strip()
        if not command:
            raise ValueError("command is empty")
        if not tasks.get(self.task_id)["auto_run"] and not self.approve(command):
            return "The user did not allow this command. Continue without it, or explain why it is needed.", \
                {"denied": True}
        timeout = max(5, min(int(_arg(args, "timeout", 120)), 600))
        code, output = self.ws.run(command, timeout)
        status = f"exit code {code}" if code is not None else "timed out"
        return f"{status}\n{output}".rstrip(), {"exit_code": code}

    def _tool_update_plan(self, args: dict) -> tuple[str, dict]:
        steps = []
        for s in args.get("steps") or []:
            if isinstance(s, str):
                steps.append({"text": s, "done": False})
            elif isinstance(s, dict) and (s.get("text") or s.get("step")):
                done = s.get("done") or str(s.get("status", "")).lower() in ("done", "completed", "complete")
                steps.append({"text": str(s.get("text") or s.get("step")), "done": bool(done)})
        fields: dict = {"plan": steps}
        if args.get("notes") is not None:
            fields["notes"] = str(args["notes"])[:4000]
        tasks.update(self.task_id, **fields)
        done = sum(s["done"] for s in steps)
        return f"Plan saved ({done}/{len(steps)} steps done).", {}


# History helpers

def _group(history: list[dict]) -> list[list[dict]]:
    """Split history into steps: a user message, or an assistant message plus its tool results."""
    units: list[list[dict]] = []
    for m in history:
        if m["role"] == "tool":
            if units and units[-1][0]["role"] == "assistant":
                units[-1].append(m)
        else:
            units.append([m])
    return units


def _shorten_args(call: dict) -> str:
    args = call_args(call)
    if call["name"] == "write_file" and "content" in args:
        args["content"] = f"[{len(str(args['content']))} characters written; read the file for its current content]"
    elif call["name"] == "edit_file":
        for key in ("old_text", "new_text"):
            if key in args:
                args[key] = _head(str(args[key]), 300)
    else:
        return call["arguments"]
    return json.dumps(args)


def _render(unit: list[dict], target: tuple[str, str], gemini: bool, short: bool) -> list[dict]:
    head = unit[0]
    if head["role"] == "user":
        return [{"role": "user", "content": _head(head["content"], 4000) if short else head["content"]}]

    calls = head.get("tool_calls") or []
    out: dict = {"role": "assistant", "content": head.get("content") or None}
    if calls:
        same_model = (head.get("provider"), head.get("model")) == target
        rendered = []
        for i, c in enumerate(calls):
            tc = {"id": c["id"], "type": "function",
                  "function": {"name": c["name"], "arguments": _shorten_args(c) if short else c["arguments"]}}
            if gemini:
                # Gemini 3 rejects tool calls without its thought signature; Google documents this
                # placeholder for history that came from another model.
                if same_model and c.get("extra"):
                    tc["extra_content"] = c["extra"]
                elif i == 0:
                    tc["extra_content"] = {"google": {"thought_signature": GEMINI_DUMMY_SIGNATURE}}
            rendered.append(tc)
        out["tool_calls"] = rendered
    elif not out["content"]:
        out["content"] = ""
    msgs = [out]

    results = {m["tool_call_id"]: m for m in unit[1:]}
    for c in calls:
        r = results.get(c["id"])
        content = r["content"] if r else "No result: the task was interrupted before this ran."
        tool = {"role": "tool", "tool_call_id": c["id"], "content": clip(content, 800) if short else content}
        if gemini:
            tool["name"] = c["name"]
        msgs.append(tool)
    return msgs


def _summarize(unit: list[dict]) -> list[str]:
    head = unit[0]
    if head["role"] == "user":
        return [f"- The user said: {_head(head['content'], 300)}"]
    lines = []
    if head.get("content"):
        lines.append(f"- You said: {_head(head['content'], 200)}")
    results = {m["tool_call_id"]: m for m in unit[1:]}
    for c in head.get("tool_calls") or []:
        r = results.get(c["id"])
        outcome = ""
        if r:
            if r.get("exit_code") is not None:
                outcome = f" -> exit code {r['exit_code']}"
            elif not r.get("ok"):
                outcome = f" -> {_head(r['content'], 120)}"
        lines.append(f"- {describe_call(c['name'], call_args(c))}{outcome}")
    return lines
