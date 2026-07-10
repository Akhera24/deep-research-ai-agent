"""
Browser-test harness for the loading screen (PLAN.md Rev 2.1, Step A3).

Boots the REAL app (uvicorn on a background thread, temp sqlite, Turnstile
TEST keys, fast SSE poll) inside this process, with the orchestrator replaced
by a test-scripted stand-in — so tests drive real routes/jobs/SSE/DB and the
real frontend, with zero research spend and no network.

Determinism: scripts pause at ("gate", threading.Event) steps; the test makes
its Playwright `expect()` assertions at a gate, then sets the event to let the
run continue. No sleep()s in test code.
"""

import asyncio
import socket
import threading
import time

import pytest
import uvicorn

from src.api import db as api_db
from src.api import jobs as jobs_mod
from src.api import routes as routes_mod


class ScriptedOrchestrator:
    """Stands in for ResearchOrchestrator; runs a per-test script.

    Steps (list assigned to ScriptedOrchestrator.script):
      ("progress", {...})        -> node-boundary progress_callback payload
      ("activity", {...})        -> activity_callback event
      ("gate", threading.Event)  -> hold until the test sets the event
      ("sleep", seconds)         -> pacing (inside the app, not the test)
      ("fail", "msg")            -> raise RuntimeError(msg)
    """

    script = []
    total_cost = 0.01           # >= PER_JOB_BUDGET_USD triggers the budget abort
    result = {
        "facts": [], "risk_flags": [], "connections": [],
        "metadata": {"coverage": {}, "iterations": 1, "duration_seconds": 0.1},
    }

    def __init__(self, max_iterations=10, enable_checkpoints=False):
        class _Cfg:
            total_cost = type(self).total_cost

        class _Client:
            config = _Cfg()

        class _Router:
            clients = {"fake": _Client()}

        self.router = _Router()

    async def research(self, query, context=None, progress_callback=None,
                       activity_callback=None):
        for step, arg in type(self).script:
            if step == "progress" and progress_callback is not None:
                await progress_callback(arg)
            elif step == "activity" and activity_callback is not None:
                await activity_callback(arg)
            elif step == "gate":
                while not arg.is_set():
                    await asyncio.sleep(0.05)
            elif step == "sleep":
                await asyncio.sleep(arg)
            elif step == "fail":
                raise RuntimeError(arg)
        return type(self).result


async def _turnstile_ok(token, ip):
    return True, "ok"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Real app on 127.0.0.1:<free-port>; yields the base URL."""
    monkeypatch.setattr(api_db.settings, "DATABASE_URL",
                        f"sqlite:///{tmp_path}/e2e.db")
    monkeypatch.setattr(api_db.settings, "ADMIN_BYPASS_TOKEN", "test-admin-token")
    monkeypatch.setattr(api_db.settings, "ENVIRONMENT", "development")
    # Cloudflare's official TEST keys — CI never touches real creds.
    monkeypatch.setattr(api_db.settings, "TURNSTILE_SITE_KEY",
                        "1x00000000000000000000AA")
    monkeypatch.setattr(api_db.settings, "TURNSTILE_SECRET_KEY",
                        "1x0000000000000000000000000000000AA")
    monkeypatch.setattr(routes_mod, "verify_turnstile", _turnstile_ok)
    monkeypatch.setattr(routes_mod, "SSE_POLL_SECONDS", 0.1)
    monkeypatch.setattr(jobs_mod, "ResearchOrchestrator", ScriptedOrchestrator)
    ScriptedOrchestrator.script = []
    ScriptedOrchestrator.total_cost = 0.01
    jobs_mod.reset_state()
    try:
        routes_mod.limiter.reset()
    except Exception:
        pass

    import src.api.main as main_mod
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(
        main_mod.app, host="127.0.0.1", port=port, log_level="warning",
    ))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not srv.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn failed to start")
        time.sleep(0.02)

    yield f"http://127.0.0.1:{port}"

    srv.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def set_script():
    def _set(script):
        ScriptedOrchestrator.script = script
    return _set


@pytest.fixture
def high_cost_orchestrator():
    """Router cost above PER_JOB_BUDGET_USD → the node-boundary budget abort
    fires on the first progress write (must run AFTER `server` in the test
    signature, since `server` resets the cost)."""
    ScriptedOrchestrator.total_cost = 5.0
    yield
    ScriptedOrchestrator.total_cost = 0.01
