from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime

from . import state
from .config import HOME, load_config
from .router import NoModelAvailable, Router

NON_CHAT = re.compile(r"(?i)(embed|whisper|tts|speech|transcri|image|vision-only|rerank|moderation|guard|audio|video|flux|sdxl)")

COLORS = {
    "ready": "\033[32m",
    "cooldown": "\033[33m",
    "exhausted": "\033[31m",
    "error": "\033[31m",
    "invalid": "\033[90m",
    "no_key": "\033[90m",
}
RESET = "\033[0m"


def _color(status: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{COLORS.get(status, '')}{text}{RESET}"


def _clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%I:%M %p").lstrip("0")


def _describe(model: dict, now: float) -> str:
    status, until = model["status"], model["until"]
    if status == "ready":
        return "ready"
    if status == "no_key":
        return "no key"
    if status == "cooldown":
        return f"cooling down ({int(until - now)}s)" if until else "cooling down"
    if status == "exhausted":
        return f"used up until {_clock(until)}" if until else "used up"
    if status == "invalid":
        return "not usable"
    if status == "error":
        return f"error, retry at {_clock(until)}" if until else "error"
    return status


def cmd_ask(args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt).strip()
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        prompt = f"{prompt}\n\n{piped}" if prompt else piped
    if not prompt:
        print("Nothing to ask. Usage: fr ask \"your question\"", file=sys.stderr)
        return 2

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})

    router = Router()
    request_id = state.start_request("ask", prompt)
    try:
        result = router.chat(messages, request_id=request_id, force=args.model)
    except NoModelAvailable as err:
        state.finish_request(request_id, ok=False, error=str(err))
        print(f"error: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        state.finish_request(request_id, ok=False, error="Cancelled")
        return 130
    state.finish_request(request_id, ok=True)
    print(result.text)
    print(f"\n[answered by {result.label}]", file=sys.stderr)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    snap = state.snapshot(load_config())
    now = snap["now"]
    for p in snap["providers"]:
        if not p["has_key"] and not args.all:
            continue
        key = "key set" if p["has_key"] else f"no key ({p['key_env']})"
        print(f"\n{p['label']}  ·  {key}  ·  resets {_clock(p['next_reset'])}")
        for m in p["models"]:
            usage = f"{m['ok']} ok / {m['limited']} limited / {m['tokens']:,} tokens" if m["attempts"] else ""
            print(f"  {_color(m['status'], '●')} {m['id']:<44} {_color(m['status'], _describe(m, now)):<34} {usage}")
    hidden = sum(1 for p in snap["providers"] if not p["has_key"])
    if hidden and not args.all:
        print(f"\n({hidden} providers without a key hidden; `fr status --all` shows them, `fr keys` lists signup links)")
    if snap["up_next"]:
        print("\nUp next: " + ", ".join(f"{u['provider']}/{u['model']}" for u in snap["up_next"][:4]))
    for r in snap["active"]:
        where = f"{r['provider']}/{r['model']}" if r["model"] else "-"
        print(f"\nRunning: [{r['kind']}] {r['activity']} ({where})")
    t = snap["totals"]
    print(f"\nToday: {t['requests']} requests, {t['ok']} answers, {t['limited']} rate limits, {t['tokens']:,} tokens")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    router = Router()
    targets = [p for p in router.providers if not args.provider or p.name == args.provider]
    if not targets:
        print(f"No provider named '{args.provider}' in providers.yaml", file=sys.stderr)
        return 2
    for p in targets:
        if not p.available:
            print(f"\n{p.label}: skipped, {p.key_env} is not set")
            continue
        try:
            ids = sorted(m.id for m in router.client(p).models.list())
        except Exception as err:  # noqa: BLE001 - report any provider failure and keep going
            print(f"\n{p.label}: could not list models ({' '.join(str(err).split())[:200]})")
            continue
        configured = {m.id for m in p.models}
        print(f"\n{p.label}: {len(ids)} models (* = in providers.yaml)")
        for model_id in ids:
            print(f"  {'*' if model_id in configured else ' '} {model_id}")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    router = Router()
    targets = [p for p in router.providers if p.discover and (not args.provider or p.name == args.provider)]
    if not targets:
        print("No providers with `discover:` match.", file=sys.stderr)
        return 2
    for p in targets:
        if not p.available:
            print(f"{p.label}: skipped, {p.key_env} is not set")
            continue
        try:
            ids = [m.id for m in router.client(p).models.list()]
        except Exception as err:  # noqa: BLE001 - report any provider failure and keep going
            print(f"{p.label}: could not list models ({' '.join(str(err).split())[:200]})")
            continue
        include = re.compile(p.discover.get("include") or ".")
        chosen = [m for m in ids if include.search(m) and not NON_CHAT.search(m)][: int(p.discover.get("limit", 8))]
        state.save_discovered(p.name, chosen)
        print(f"{p.label}: saved {len(chosen)} of {len(ids)} models: {', '.join(chosen)}")
    return 0


def cmd_keys(args: argparse.Namespace) -> int:
    providers = load_config().providers
    have = sum(1 for p in providers if p.available)
    print(f"{have} of {len(providers)} providers ready. Add keys to {HOME / '.env'}\n")
    for p in providers:
        if p.available:
            mark = _color("ready", "✓")
        else:
            mark = _color("no_key", "·")
        how = f"set {p.key_env}=1 to opt in" if p.keyless else p.key_env
        trust = "  [third-party]" if p.trust else ""
        print(f" {mark} {p.label:<30} {how:<30} {p.signup}{trust}")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    provider, _, model = (args.target or "").partition("/")
    cleared = state.clear_status(provider or None, model or None)
    print(f"Cleared {cleared} status entr{'y' if cleared == 1 else 'ies'}.")
    return 0


def _print_agent_message(m: dict) -> None:
    from .agent import call_args, describe_call

    if m["role"] == "note":
        level = {"error": "error", "warning": "cooldown", "model": "invalid"}.get(m.get("level", ""), "invalid")
        print(_color(level, f"· {m['content']}"))
    elif m["role"] == "assistant":
        if m.get("content"):
            print(f"\n{m['content']}")
        for call in m.get("tool_calls") or []:
            print(_color("invalid", f"  → {describe_call(call['name'], call_args(call))}"))
    elif m["role"] == "tool" and (not m.get("ok") or m.get("exit_code") not in (None, 0)):
        first = (m.get("content") or "").splitlines()[0:1]
        print(_color("error", f"    {first[0] if first else 'failed'}"))


def cmd_code(args: argparse.Namespace) -> int:
    from . import tasks
    from .agent import Agent
    from .workspace import Workspace, WorkspaceError

    try:
        root = str(Workspace(args.dir).root)
    except WorkspaceError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    task_id = args.task
    if task_id and not tasks.get(task_id):
        print(f"error: no task {task_id}", file=sys.stderr)
        return 2

    def approve(command: str) -> bool:
        try:
            return input(f"  Run `{command}`? [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:
            return False

    prompt = " ".join(args.prompt).strip()
    print(f"Working in {root}. Empty line or Ctrl+D to quit.")
    while True:
        if not prompt:
            try:
                prompt = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not prompt:
                return 0
        if task_id is None:
            task_id = tasks.create(root, prompt.splitlines()[0], auto_run=args.yes)
            print(_color("invalid", f"· Task {task_id} (continue it later with: fr code --task {task_id})"))
        elif args.yes:
            tasks.update(task_id, auto_run=True)
        tasks.add_message(task_id, "user", {"content": prompt})
        Agent(task_id, approve=approve, on_message=_print_agent_message).run()
        prompt = ""


def cmd_dashboard(args: argparse.Namespace) -> int:
    import uvicorn

    url = f"http://127.0.0.1:{args.port}"
    print(f"Dashboard running at {url}  (workspace: {url}/workspace, config: {HOME})")
    if args.open:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run("freeroute.dashboard:app", host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fr", description="Route prompts across free LLM APIs.")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask a question through the fallback chain")
    ask.add_argument("prompt", nargs="*", help="The question (stdin is appended if piped)")
    ask.add_argument("-m", "--model", help="Force a provider, model, or provider/model")
    ask.add_argument("-s", "--system", help="Optional system prompt")
    ask.set_defaults(func=cmd_ask)

    code = sub.add_parser("code", help="Coding agent that edits files in a folder, switching models as quotas run out")
    code.add_argument("prompt", nargs="*", help="The task (omit to type it interactively)")
    code.add_argument("-C", "--dir", default=".", help="Project folder (default: current folder)")
    code.add_argument("-t", "--task", type=int, help="Continue an earlier task by its number")
    code.add_argument("-y", "--yes", action="store_true", help="Run shell commands without asking")
    code.set_defaults(func=cmd_code)

    status = sub.add_parser("status", help="Show providers and models with their current state")
    status.add_argument("--all", action="store_true", help="Include providers without a key")
    status.set_defaults(func=cmd_status)

    keys = sub.add_parser("keys", help="List every provider, its signup link, and whether a key is set")
    keys.set_defaults(func=cmd_keys)

    sync = sub.add_parser("sync", help="Fetch model lists for providers marked `discover:`")
    sync.add_argument("provider", nargs="?", help="Only this provider")
    sync.set_defaults(func=cmd_sync)

    models = sub.add_parser("models", help="List the model IDs each provider's key can reach")
    models.add_argument("provider", nargs="?", help="Only this provider")
    models.set_defaults(func=cmd_models)

    reset = sub.add_parser("reset", help="Clear used-up / cooldown marks")
    reset.add_argument("target", nargs="?", help="provider or provider/model (default: everything)")
    reset.set_defaults(func=cmd_reset)

    dash = sub.add_parser("dashboard", help="Start the live web dashboard")
    dash.add_argument("--port", type=int, default=4000)
    dash.add_argument("--open", action="store_true", help="Open it in the browser")
    dash.set_defaults(func=cmd_dashboard)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
