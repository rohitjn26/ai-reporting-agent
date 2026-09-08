"""
Singleton localhost HTTP server that serves the most recent chart HTML.
Starts once on import; port 8080 by default.
"""
import threading, http.server, webbrowser, os

_current_html: str = "<h1>No chart yet</h1>"
_started = False
_lock = threading.Lock()
_PORT: int = 0  # resolved on first _ensure_started() call


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = _current_html.encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def log_message(self, *_):
        pass


def _ensure_started():
    global _started, _PORT
    with _lock:
        if _started:
            return
        preferred = int(os.environ.get("CHART_PORT", "8080"))
        # Try the preferred port first, then let the OS pick a free one.
        for port in (preferred, 0):
            try:
                server = http.server.HTTPServer(("", port), _Handler)
                _PORT = server.server_address[1]  # actual port (matters when port=0)
                break
            except OSError:
                if port == 0:
                    raise  # OS couldn't find a free port — give up
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        _started = True


def serve_chart(html: str, open_browser: bool = True) -> str:
    """
    Update the served chart HTML and return the localhost URL.
    Set env var CHART_OPEN_BROWSER=false to suppress auto-open (used by web UI).
    """
    global _current_html
    _ensure_started()
    _current_html = html
    url = f"http://localhost:{_PORT}"
    should_open = open_browser and os.environ.get("CHART_OPEN_BROWSER", "true").lower() != "false"
    if should_open:
        webbrowser.open(url)
    return url
