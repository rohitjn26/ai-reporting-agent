"""
Shared pytest fixtures.

Import notes
------------
* Agent modules (`chart.*`, `graph.*`) import fine because `pytest.ini` puts the
  `agent/` directory on the path.
* `mcp/*.py` and `library/*.py` are loaded by *file path* under unique module
  names. This avoids adding the repo root to `sys.path`, which would let the local
  `mcp/` directory shadow the installed `mcp` SDK that those modules import.
"""
import importlib.util
import os
import pathlib
import socket
import threading
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Charts must never try to pop open a browser during tests.
os.environ.setdefault("CHART_OPEN_BROWSER", "false")


def _load_by_path(module_name: str, rel_path: str):
    """Load a module from a file path under a unique name (no sys.path changes)."""
    spec = importlib.util.spec_from_file_location(module_name, ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Path-loaded modules under test ────────────────────────────────────────────

@pytest.fixture(scope="session")
def cube_server_mod():
    """The `mcp/cube_server.py` module (for its pure helpers)."""
    return _load_by_path("cube_server_under_test", "mcp/cube_server.py")


@pytest.fixture(scope="session")
def library_server_mod():
    """The `mcp/library_server.py` module (validation helpers + MCP tools)."""
    return _load_by_path("library_server_under_test", "mcp/library_server.py")


# ── End-to-end: disposable Postgres + the library API running for real ────────

@pytest.fixture(scope="session")
def live_library_server(library_server_mod):
    """
    Spin up a throwaway Postgres, run the FastAPI library app against it in a
    background uvicorn thread, and point the library MCP tools at that server.

    Yields the configured `library_server` module. Skips the whole test if Docker
    or testcontainers is unavailable, so unit tests still run on their own.
    """
    try:
        from testcontainers.postgres import PostgresContainer
    except Exception as exc:  # pragma: no cover - import guard
        pytest.skip(f"testcontainers not available: {exc}")

    try:
        postgres = PostgresContainer("postgres:16-alpine")
        postgres.start()
    except Exception as exc:  # Docker daemon down, image pull failed, etc.
        pytest.skip(f"could not start Postgres container (is Docker running?): {exc}")

    server = None
    thread = None
    try:
        # Use the pure-Python pg8000 driver so no C compiler / libpq is needed.
        raw_url = postgres.get_connection_url()  # postgresql+psycopg2://...
        db_url = raw_url.replace("+psycopg2", "+pg8000")
        os.environ["DATABASE_URL"] = db_url

        # Load the app AFTER DATABASE_URL is set (engine is built at import time).
        app_mod = _load_by_path("library_app_under_test", "library/app.py")

        import uvicorn

        port = _free_port()
        config = uvicorn.Config(app_mod.app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        base_url = f"http://127.0.0.1:{port}"
        _wait_for_health(server, base_url)

        # Point the MCP library tools at our live server.
        library_server_mod.LIBRARY_URL = base_url

        yield library_server_mod
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=5)
        postgres.stop()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(server, base_url: str, timeout: float = 30.0) -> None:
    """Block until the app answers /health (its startup hook creates the table)."""
    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        if getattr(server, "started", False):
            try:
                r = httpx.get(f"{base_url}/health", timeout=2)
                if r.status_code == 200:
                    return
            except Exception:
                pass
        time.sleep(0.1)
    raise RuntimeError("library API did not become healthy in time")
