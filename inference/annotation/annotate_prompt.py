import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np
from decord import VideoReader

_HERE = Path(__file__).parent


_STATIC: dict[str, tuple[str, Path]] = {
    "/":           ("text/html; charset=utf-8", _HERE / "annotate.html"),
    "/style.css":  ("text/css",                 _HERE / "annotate.css"),
    "/annotate.js":("application/javascript",   _HERE / "annotate.js"),
}

def _frame_jpeg(vr: VideoReader, idx: int) -> bytes:
    idx = max(0, min(idx, len(vr) - 1))
    rgb = vr[idx].asnumpy()[..., :3]
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return bytes(buf)

class _Handler(BaseHTTPRequestHandler):
    vr:          VideoReader
    video_path:  Path
    output_json: Path
    done_event:  threading.Event
    server_ref:  HTTPServer

    def log_message(self, fmt, *args):
        pass  

    def _respond(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in _STATIC:
            content_type, fpath = _STATIC[self.path]
            self._respond(200, content_type, fpath.read_bytes())

        elif self.path.startswith("/frame/"):
            try:
                idx = int(self.path.split("/frame/")[1].split("?")[0])
                self._respond(200, "image/jpeg", _frame_jpeg(self.vr, idx))
            except Exception as exc:
                self._respond(500, "text/plain", str(exc).encode())

        else:
            self._respond(404, "text/plain", b"Not found")

    def do_POST(self):
        if self.path != "/save":
            self._respond(404, "text/plain", b"Not found")
            return

        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        pts  = [[int(x), int(y)] for x, y in body["points"]]
        fidx = int(body.get("frame_index", 0))

        payload = {
            "video_path":  str(self.video_path.resolve()),
            "frame_index": fidx,
            "groups":      {"group_0": pts},
        }
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        self.output_json.write_text(json.dumps(payload, indent=2))

        msg = f"Saved {len(pts)} point(s) (frame {fidx}) → {self.output_json}"
        print(msg)
        self._respond(200, "text/plain", msg.encode())
        threading.Thread(target=self._shutdown, daemon=True).start()

    def _shutdown(self):
        import time
        time.sleep(0.4)
        self.done_event.set()
        self.server_ref.shutdown()

def annotate(video_path: Path, output_json: Path,
             host: str = "127.0.0.1", port: int = 7860,
             open_browser: bool = True) -> bool:
    """Start the annotator server. Blocks until the user saves or Ctrl-C."""
    vr = VideoReader(str(video_path))
    if len(vr) == 0:
        raise ValueError(f"No frames in {video_path}")

    done_event = threading.Event()
    server_box: list[HTTPServer] = []

    def make_handler():
        class H(_Handler):
            @property
            def server_ref(self):
                return server_box[0]
        H.vr          = vr
        H.video_path  = video_path
        H.output_json = output_json
        H.done_event  = done_event
        return H

    server = HTTPServer((host, port), make_handler())
    server_box.append(server)

    url = f"http://{host}:{port}"
    print(f"Video : {video_path}")
    print(f"Output: {output_json}")
    print(f"Open  : {url}")
    print("Click on the target object, then press Save (or Enter).")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nInterrupted — no file written.")
        return False

    return done_event.is_set()


def main():
    p = argparse.ArgumentParser(
        description="Annotate a prompt point on a video frame (offline, no Gradio)."
    )
    p.add_argument("--input-video", required=True, help="Path to input video file.")
    p.add_argument("--output-json", default=None,
                   help="Output JSON path. Default: <video_stem>.points.json alongside the video.")
    p.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1).")
    p.add_argument("--port", type=int, default=7860, help="Server port (default: 7860).")
    p.add_argument("--no-browser", action="store_true",
                   help="Do not try to open a browser automatically.")
    args = p.parse_args()

    video = Path(args.input_video).expanduser().resolve()
    out   = (video.with_suffix(".points.json") if args.output_json is None
             else Path(args.output_json).expanduser().resolve())

    annotate(video, out, host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
