"""
Local server for generate.html: accepts a YouTube URL from the browser,
runs the same download+transcribe pipeline as yt_sync.py in a background
thread, and lets the page poll job progress until the audio+transcript
are ready in output/.

A browser page on its own can't call yt-dlp/ffmpeg/Whisper (no filesystem
or process access from JS) -- this server is what actually does the work;
the HTML is just a thin UI in front of it.

Usage: python server.py
Then open http://127.0.0.1:8765/ in a browser.
"""
import json
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

import yt_sync

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
PORT = 8765

jobs = {}
jobs_lock = threading.Lock()


def run_job(job_id, url, model_size, language):
    def set_state(**kwargs):
        with jobs_lock:
            jobs[job_id].update(kwargs)

    t0 = time.time()
    try:
        result = yt_sync.process_video(
            url, model_size, language=language, out_dir=OUTPUT_DIR,
            on_progress=lambda msg: set_state(stage="working", message=msg)
        )
        sync_data = result["sync_data"]
        set_state(
            stage="done", message="完成！", done=True,
            result={
                "title": result["title"],
                "audio_file": result["audio_path"].name,
                "json_file": result["json_path"].name,
                "sentences": len(sync_data["sentences"]),
                "duration": sync_data["duration"],
                "used_captions": result["used_captions"],
                "elapsed_seconds": time.time() - t0,
            }
        )
    except Exception as e:
        set_state(stage="error", message=str(e), done=True, error=f"{e}\n{traceback.format_exc()}")


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path, content_type):
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self._send_json({"error": "not found"}, status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_file(BASE_DIR / "generate.html", "text/html; charset=utf-8")
        elif parsed.path == "/api/languages":
            self._send_json(yt_sync.LANGUAGES)
        elif parsed.path == "/api/status":
            qs = parse_qs(parsed.query)
            job_id = (qs.get("job_id") or [""])[0]
            with jobs_lock:
                job = jobs.get(job_id)
            if job is None:
                self._send_json({"error": "job not found"}, status=404)
            else:
                self._send_json(job)
        elif parsed.path.startswith("/output/"):
            fname = unquote(parsed.path[len("/output/"):])
            fpath = (OUTPUT_DIR / fname).resolve()
            if ".." in fname or not fpath.is_relative_to(OUTPUT_DIR.resolve()) or not fpath.exists():
                self._send_json({"error": "not found"}, status=404)
                return
            ctype = "audio/mpeg" if fpath.suffix == ".mp3" else "application/json; charset=utf-8"
            self._serve_file(fpath, ctype)
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/generate":
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except Exception:
            self._send_json({"error": "invalid JSON"}, status=400)
            return
        url = (payload.get("url") or "").strip()
        model_size = (payload.get("model") or "small").strip()
        language = (payload.get("language") or yt_sync.DEFAULT_LANGUAGE).strip()
        if not url:
            self._send_json({"error": "缺少 url"}, status=400)
            return
        if language not in yt_sync.LANGUAGES:
            self._send_json({"error": f"不支援的語言：{language}"}, status=400)
            return

        job_id = uuid.uuid4().hex[:12]
        with jobs_lock:
            jobs[job_id] = {"stage": "queued", "message": "已加入佇列...", "done": False, "error": None, "result": None}
        threading.Thread(target=run_job, args=(job_id, url, model_size, language), daemon=True).start()
        self._send_json({"job_id": job_id})

    def log_message(self, format, *args):
        pass  # job status polling already surfaces progress; keep console quiet


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Server running at http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
