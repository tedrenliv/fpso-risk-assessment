import json
import threading
import re
from flask import Flask, Response, request, jsonify, send_from_directory
from core.stream_queue import get_queue, reset as reset_queue, install as install_stdout, uninstall as uninstall_stdout
from core.hitl_bridge import set_decision, reset as reset_hitl
from agents.orchestrator import run_pipeline

app = Flask(__name__)

_pipeline_running = False
_pipeline_result = None
_pipeline_error = None

_TAG_RE = re.compile(r'^\[([^\]]+)\]')


class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)


@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")


@app.route("/run", methods=["POST"])
def run():
    global _pipeline_running, _pipeline_result, _pipeline_error
    if _pipeline_running:
        return jsonify({"error": "Pipeline already running"}), 409

    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    run_feedback = bool(data.get("run_feedback", False))
    forced_skills = data.get("forced_skills") or None  # list of domain IDs or None

    if not query:
        return jsonify({"error": "Query is required"}), 400

    _pipeline_running = True
    _pipeline_result = None
    _pipeline_error = None
    reset_queue()
    reset_hitl()
    install_stdout()

    def _run():
        global _pipeline_running, _pipeline_result, _pipeline_error
        try:
            result = run_pipeline(query, run_feedback=run_feedback,
                                  forced_skills=forced_skills)
            _pipeline_result = result
        except Exception as exc:
            _pipeline_error = str(exc)
        finally:
            uninstall_stdout()
            _pipeline_running = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started"}), 200


@app.route("/stream")
def stream():
    def _generate():
        q = get_queue()
        while True:
            try:
                msg = q.get(timeout=0.2)
            except Exception:
                msg = None

            if msg is not None:
                if msg.startswith("__HITL__:"):
                    payload = msg[len("__HITL__:"):]
                    yield f"event: hitl\ndata: {payload}\n\n"
                else:
                    m = _TAG_RE.match(msg)
                    tag = f"[{m.group(1)}]" if m else "[SYS]"
                    msg_type = "ok" if "→" in msg else "info"
                    data = json.dumps({"msg": msg, "tag": tag, "type": msg_type})
                    yield f"data: {data}\n\n"

            if _pipeline_result is not None:
                # Drain any remaining queued messages before sending result
                while True:
                    try:
                        remaining = q.get_nowait()
                        if not remaining.startswith("__HITL__:"):
                            m = _TAG_RE.match(remaining)
                            tag = f"[{m.group(1)}]" if m else "[SYS]"
                            msg_type = "ok" if "→" in remaining else "info"
                            data = json.dumps({"msg": remaining, "tag": tag, "type": msg_type})
                            yield f"data: {data}\n\n"
                    except Exception:
                        break
                payload = json.dumps(_pipeline_result, cls=_SafeEncoder)
                yield f"event: result\ndata: {payload}\n\n"
                break

            if _pipeline_error is not None:
                payload = json.dumps({"message": _pipeline_error})
                yield f"event: error\ndata: {payload}\n\n"
                break

            if not _pipeline_running and _pipeline_result is None and _pipeline_error is None:
                yield f"event: error\ndata: {json.dumps({'message': 'Pipeline ended unexpectedly'})}\n\n"
                break

            yield ": keepalive\n\n"

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/hitl", methods=["POST"])
def hitl():
    data = request.get_json(force=True) or {}
    set_decision(data.get("decision", "approved"), data.get("note", ""))
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    from main import seed_knowledge_base
    seed_knowledge_base()
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
