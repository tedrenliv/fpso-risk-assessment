# Dashboard Live Query Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire `dashboard.html` to the real Python pipeline so users can type queries and see live results stream in real-time via SSE, with HITL buttons that actually control the pipeline.

**Architecture:** Flask server (`server.py`) serves the dashboard and exposes `/run`, `/stream`, `/hitl`. A `QueueWriter` redirects `sys.stdout` so all agent `print()` calls flow into a `queue.Queue` which the SSE endpoint reads. The validator pauses on a `threading.Event` instead of `input()`, unblocked by `POST /hitl`.

**Tech Stack:** Flask, Python `threading`, `queue`, Server-Sent Events (SSE), existing Anthropic SDK + ChromaDB pipeline.

---

### Task 1: Add Flask to requirements

**Files:**
- Modify: `requirements.txt`

**Step 1: Add flask**

Append to `requirements.txt`:
```
flask>=3.0.0
```

**Step 2: Install**

```bash
pip install flask
```
Expected: `Successfully installed flask-...`

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add flask dependency for web server"
```

---

### Task 2: Create `core/stream_queue.py`

**Files:**
- Create: `core/stream_queue.py`
- Create: `tests/test_stream_queue.py`

**Step 1: Write the failing test**

Create `tests/test_stream_queue.py`:
```python
import sys
from core.stream_queue import get_queue, reset, install, uninstall


def test_print_captured_to_queue():
    reset()
    install()
    print("hello world")
    uninstall()
    q = get_queue()
    assert not q.empty()
    msg = q.get_nowait()
    assert "hello world" in msg


def test_stdout_still_writes_through():
    """Original stdout must still receive output."""
    reset()
    install()
    import io
    # After install, sys.stdout is our writer; its _orig is the real stdout
    assert hasattr(sys.stdout, '_orig')
    uninstall()


def test_reset_clears_queue():
    reset()
    install()
    print("some message")
    uninstall()
    reset()
    assert get_queue().empty()


def test_install_idempotent():
    """Calling install twice should not double-wrap stdout."""
    reset()
    install()
    writer_after_first = sys.stdout
    install()  # second call — should be a no-op
    assert sys.stdout is writer_after_first
    uninstall()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_stream_queue.py -v
```
Expected: `ImportError` or `ModuleNotFoundError` — `core.stream_queue` does not exist yet.

**Step 3: Write implementation**

Create `core/stream_queue.py`:
```python
import queue
import sys

_q: queue.Queue = queue.Queue()
_orig_stdout = None


def get_queue() -> queue.Queue:
    return _q


def reset() -> None:
    while not _q.empty():
        try:
            _q.get_nowait()
        except queue.Empty:
            break


class _QueueWriter:
    def __init__(self, original):
        self._orig = original

    def write(self, text: str) -> None:
        stripped = text.rstrip("\n")
        if stripped.strip():
            _q.put(stripped)
        self._orig.write(text)

    def flush(self) -> None:
        self._orig.flush()


def install() -> None:
    global _orig_stdout
    if _orig_stdout is None:
        _orig_stdout = sys.stdout
        sys.stdout = _QueueWriter(_orig_stdout)


def uninstall() -> None:
    global _orig_stdout
    if _orig_stdout is not None:
        sys.stdout = _orig_stdout
        _orig_stdout = None
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_stream_queue.py -v
```
Expected: 4 PASSED.

**Step 5: Commit**

```bash
git add core/stream_queue.py tests/test_stream_queue.py
git commit -m "feat: add stream_queue to capture stdout into SSE queue"
```

---

### Task 3: Create `core/hitl_bridge.py`

**Files:**
- Create: `core/hitl_bridge.py`
- Create: `tests/test_hitl_bridge.py`

**Step 1: Write the failing test**

Create `tests/test_hitl_bridge.py`:
```python
import threading
from core.hitl_bridge import wait_for_decision, set_decision, reset


def test_set_before_wait_returns_immediately():
    reset()
    set_decision("approved", "looks good")
    decision, note = wait_for_decision()
    assert decision == "approved"
    assert note == "looks good"


def test_wait_blocks_until_set():
    reset()

    def _setter():
        import time
        time.sleep(0.05)
        set_decision("skipped", "")

    t = threading.Thread(target=_setter)
    t.start()
    decision, note = wait_for_decision()
    t.join()
    assert decision == "skipped"


def test_reset_clears_event():
    reset()
    set_decision("rejected", "bad")
    reset()
    # After reset, wait_for_decision should block — verify event is cleared
    from core import hitl_bridge
    assert not hitl_bridge._event.is_set()


def test_default_decision_after_reset():
    reset()
    from core import hitl_bridge
    assert hitl_bridge._decision["value"] == "approved"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_hitl_bridge.py -v
```
Expected: `ImportError` — module does not exist yet.

**Step 3: Write implementation**

Create `core/hitl_bridge.py`:
```python
import threading

_event = threading.Event()
_decision: dict = {"value": "approved", "note": ""}


def wait_for_decision() -> tuple:
    """Block until set_decision() is called. Returns (decision, note)."""
    _event.wait()
    return _decision["value"], _decision["note"]


def set_decision(decision: str, note: str = "") -> None:
    """Called by POST /hitl to unblock the waiting pipeline."""
    _decision["value"] = decision
    _decision["note"] = note
    _event.set()


def reset() -> None:
    _event.clear()
    _decision["value"] = "approved"
    _decision["note"] = ""
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_hitl_bridge.py -v
```
Expected: 4 PASSED.

**Step 5: Commit**

```bash
git add core/hitl_bridge.py tests/test_hitl_bridge.py
git commit -m "feat: add hitl_bridge for web-based expert validation"
```

---

### Task 4: Modify `agents/validator_agent.py`

Replace the terminal `input()` HITL block with bridge calls. Preserve terminal fallback when running outside the web server (when `stream_queue` is not installed).

**Files:**
- Modify: `agents/validator_agent.py:46-73`
- Create: `tests/test_validator_agent.py`

**Step 1: Write the failing test**

Create `tests/test_validator_agent.py`:
```python
from unittest.mock import patch, MagicMock
from core import hitl_bridge, stream_queue


def _make_synthesis():
    return {
        "combined_bpa": {"Low": 0.1, "Medium": 0.2, "High": 0.7},
        "dominant_factor_name": "EH",
        "dominant_belief_high": 0.7,
        "risk_ranking": [
            {"rank": 1, "domain_name": "Environmental Hazard",
             "risk_score": 0.83, "high_belief": 0.70}
        ],
        "belief_intervals": {},
    }


def _make_mock_response():
    mock = MagicMock()
    mock.content = [MagicMock(text=(
        '{"validation_summary": "Risk is high", "flags": [], '
        '"recommendation": "approve", "confidence_note": ""}'
    ))]
    return mock


def test_validator_uses_hitl_bridge_in_web_mode():
    """When stream_queue is installed (web mode), validator uses hitl_bridge."""
    stream_queue.reset()
    stream_queue.install()
    hitl_bridge.reset()
    hitl_bridge.set_decision("approved", "")  # pre-set so it doesn't block

    with patch("agents.validator_agent.client.messages.create",
               return_value=_make_mock_response()):
        from agents import validator_agent
        result = validator_agent.run_validator(_make_synthesis(), "test query")

    stream_queue.uninstall()
    assert result["validation"]["expert_decision"] == "approved"


def test_validator_falls_back_to_input_in_terminal_mode(monkeypatch):
    """When stream_queue is NOT installed (terminal mode), validator uses input()."""
    stream_queue.uninstall()  # ensure terminal mode

    monkeypatch.setattr("builtins.input", lambda _: "A")

    with patch("agents.validator_agent.client.messages.create",
               return_value=_make_mock_response()):
        from agents import validator_agent
        result = validator_agent.run_validator(_make_synthesis(), "test query")

    assert result["validation"]["expert_decision"] == "approved"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_validator_agent.py -v
```
Expected: Both tests fail because validator still uses `input()` in all modes.

**Step 3: Modify `agents/validator_agent.py`**

Replace the HITL block (lines 46–73 — from the `# ─── HUMAN-IN-THE-LOOP PAUSE ───` comment through the `expert_note` assignment):

Old block to replace:
```python
    # ─── HUMAN-IN-THE-LOOP PAUSE ───
    print("\n" + "="*60)
    print("EXPERT VALIDATION CHECKPOINT")
    print("="*60)
    print(f"\nValidation Summary: {validation_report.get('validation_summary', '')}")
    print(f"\nFlags:")
    for flag in validation_report.get("flags", []):
        print(f"  ⚠  {flag}")
    print(f"\nAI Recommendation: {validation_report.get('recommendation', 'review').upper()}")
    print(f"\nRisk Ranking:")
    for item in synthesis_output.get("risk_ranking", []):
        print(f"  #{item['rank']} {item['domain_name']}: score={item['risk_score']:.3f}, High belief={item['high_belief']:.3f}")
    print("\n" + "-"*60)
    print("Options: [A]pprove  [R]eject  [M]odify weights  [S]kip")
    expert_input = input("Expert decision: ").strip().upper()
    # ──────────────────────────────

    expert_decision = {
        "A": "approved",
        "R": "rejected",
        "M": "modify_requested",
        "S": "skipped",
    }.get(expert_input, "approved")

    expert_note = ""
    if expert_decision in ("rejected", "modify_requested"):
        expert_note = input("Please provide reason/instructions: ").strip()
```

Replace with:
```python
    # ─── HUMAN-IN-THE-LOOP PAUSE ───
    print("\n" + "="*60)
    print("EXPERT VALIDATION CHECKPOINT")
    print("="*60)
    print(f"\nValidation Summary: {validation_report.get('validation_summary', '')}")
    print(f"\nFlags:")
    for flag in validation_report.get("flags", []):
        print(f"  ⚠  {flag}")
    print(f"\nAI Recommendation: {validation_report.get('recommendation', 'review').upper()}")
    print(f"\nRisk Ranking:")
    for item in synthesis_output.get("risk_ranking", []):
        print(f"  #{item['rank']} {item['domain_name']}: score={item['risk_score']:.3f}, High belief={item['high_belief']:.3f}")
    print("\n" + "-"*60)

    from core import stream_queue as _sq
    if _sq._orig_stdout is not None:
        # Web mode: signal the SSE stream and wait for browser HITL
        import json as _json
        from core import hitl_bridge as _hb
        _hitl_payload = {
            "summary": validation_report.get("validation_summary", ""),
            "flags": validation_report.get("flags", []),
            "recommendation": validation_report.get("recommendation", "review"),
            "ranking": synthesis_output.get("risk_ranking", []),
        }
        _sq.get_queue().put(f"__HITL__:{_json.dumps(_hitl_payload)}")
        expert_decision, expert_note = _hb.wait_for_decision()
    else:
        # Terminal mode: original input() flow
        print("Options: [A]pprove  [R]eject  [M]odify weights  [S]kip")
        expert_input = input("Expert decision: ").strip().upper()
        expert_decision = {
            "A": "approved",
            "R": "rejected",
            "M": "modify_requested",
            "S": "skipped",
        }.get(expert_input, "approved")
        expert_note = ""
        if expert_decision in ("rejected", "modify_requested"):
            expert_note = input("Please provide reason/instructions: ").strip()
    # ──────────────────────────────
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_validator_agent.py -v
```
Expected: 2 PASSED.

**Step 5: Run full test suite to check for regressions**

```bash
pytest tests/ -m "not integration" -v
```
Expected: All existing tests still pass.

**Step 6: Commit**

```bash
git add agents/validator_agent.py tests/test_validator_agent.py
git commit -m "feat: validator uses hitl_bridge in web mode, falls back to input() in terminal"
```

---

### Task 5: Create `server.py`

**Files:**
- Create: `server.py`
- Create: `tests/test_server.py`

**Step 1: Write the failing tests**

Create `tests/test_server.py`:
```python
import json
import pytest
import server  # import before patching globals


@pytest.fixture(autouse=True)
def reset_state():
    server._pipeline_running = False
    server._pipeline_result = None
    server._pipeline_error = None
    from core import hitl_bridge, stream_queue
    hitl_bridge.reset()
    stream_queue.reset()
    yield


@pytest.fixture
def client():
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


def test_run_rejects_empty_query(client):
    resp = client.post("/run", json={"query": ""})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_run_rejects_concurrent(client):
    server._pipeline_running = True
    resp = client.post("/run", json={"query": "test query"})
    assert resp.status_code == 409


def test_run_starts_pipeline(client, monkeypatch):
    started = []

    def fake_pipeline(query, run_feedback=False):
        started.append(query)
        return {"risk_rankings": [], "risk_profile": {}, "traceability": {}, "skill_trace": {}}

    monkeypatch.setattr("server.run_pipeline", fake_pipeline)
    resp = client.post("/run", json={"query": "assess blade corrosion"})
    assert resp.status_code == 200
    import time; time.sleep(0.1)  # let thread run
    assert "assess blade corrosion" in started


def test_hitl_sets_decision(client):
    from core import hitl_bridge
    hitl_bridge.reset()
    resp = client.post("/hitl", json={"decision": "approved", "note": "looks good"})
    assert resp.status_code == 200
    decision, note = hitl_bridge.wait_for_decision()
    assert decision == "approved"
    assert note == "looks good"


def test_index_serves_dashboard(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"FPSO" in resp.data or b"Risk" in resp.data
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_server.py -v
```
Expected: `ImportError` — `server` module does not exist.

**Step 3: Write implementation**

Create `server.py`:
```python
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
            result = run_pipeline(query, run_feedback=run_feedback)
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_server.py -v
```
Expected: 5 PASSED.

**Step 5: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: add Flask server with SSE streaming and HITL endpoints"
```

---

### Task 6: Update `dashboard.html` — replace JS simulation

This task replaces the hardcoded simulation with real SSE + fetch calls. All changes are in the `<script>` block (lines 593–916).

**Files:**
- Modify: `dashboard.html`

**Step 1: Remove hardcoded `RESULTS` constant**

In `dashboard.html`, find and remove lines 600–606:
```javascript
const RESULTS = {
    SI:   { score:.78, bel:.62, pl:.88, label:'HIGH'   },
    MM:   { score:.71, bel:.55, pl:.82, label:'HIGH'   },
    EH:   { score:.83, bel:.70, pl:.91, label:'HIGH'   },
    HF:   { score:.44, bel:.30, pl:.58, label:'MEDIUM' },
    SysI: { score:.52, bel:.38, pl:.65, label:'MEDIUM' }
};
```
Replace with nothing (delete it).

**Step 2: Add node-state helper and tag-seen tracker after the `DOM_LABELS` line**

Find:
```javascript
const DOM_LABELS = { SI:'Structural Integrity', MM:'Maintenance Mgmt', EH:'Environmental Hazard', HF:'Human Factors', SysI:'System Integration' };
```
Add immediately after it:
```javascript

// Map log tags → node IDs for live pipeline visualization
const _TAG_NODE = {
    '[L2]':        {activate:'rag',        done:'orchestrator'},
    '[L3]':        {activate:'router',     done:'rag'},
    '[L4]':        {activate:null,         done:'router'},
    '[L5]':        {activate:'synthesis',  done:null},
    '[Validator]': {activate:'validator',  done:'synthesis'},
    '[L6]':        {activate:'output',     done:'validator'},
};
const _tagsSeen = new Set();

function updateNodeFromLog(tag) {
    if (_tagsSeen.has(tag)) return;
    _tagsSeen.add(tag);
    const entry = _TAG_NODE[tag];
    if (!entry) return;
    if (entry.done)     nstate(entry.done,     'done');
    if (entry.activate) nstate(entry.activate, 'active');
    if (tag === '[L4]') {
        ['si','mm','eh','hf','sysi'].forEach(n => nstate(n, 'active'));
        ['fa-rtr-si','fa-rtr-mm','fa-rtr-eh','fa-rtr-hf','fa-rtr-sysi'].forEach(f => fstate(f, true));
    }
}
```

**Step 3: Replace `runPipeline()`**

Find the entire function from `async function runPipeline(){` to its closing `}` (lines 718–805) and replace with:

```javascript
async function runPipeline(){
    if(running) return;
    const query = document.querySelector('.query-textarea').value.trim();
    if(!query){ log('Please enter a query.','warn','[UI]'); return; }

    running=true;
    t0=null;
    _tagsSeen.clear();
    document.getElementById('runBtn').disabled=true;
    document.getElementById('hitlBanner').classList.remove('show');
    document.getElementById('rankScroll').innerHTML='<div style="font-family:var(--font-mono);font-size:10px;color:var(--text-2);text-align:center;padding:10px 0">Running analysis...</div>';
    document.getElementById('traceOut').textContent='—';

    ['orchestrator','rag','router','si','mm','eh','hf','sysi','synthesis','validator','output'].forEach(n=>nstate(n,'idle'));
    document.querySelectorAll('.fanim').forEach(f=>f.classList.remove('on'));
    updateRadar({SI:0,MM:0,EH:0,HF:0,SysI:0});

    startTimer();
    orchState('running');
    nstate('orchestrator','active');
    document.getElementById('pipeMsg').innerHTML='Pipeline executing...';

    // Start pipeline on server
    const runResp = await fetch('/run',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({query, run_feedback:false})
    });
    if(!runResp.ok){
        const err = await runResp.json();
        log(err.error||'Failed to start pipeline','warn','[ERR]');
        orchState('idle'); stopTimer(); running=false;
        document.getElementById('runBtn').disabled=false;
        return;
    }

    // Stream events
    const evtSource = new EventSource('/stream');

    evtSource.onmessage = (e) => {
        const d = JSON.parse(e.data);
        log(d.msg, d.type||'info', d.tag||'[SYS]');
        updateNodeFromLog(d.tag||'');
    };

    evtSource.addEventListener('hitl', (e) => {
        const d = JSON.parse(e.data);
        stopTimer();
        orchState('waiting');
        nstate('validator','active');
        document.getElementById('hitlBanner').classList.add('show');
        document.getElementById('pipeMsg').innerHTML='<em>Awaiting expert validation...</em>';
        log(d.summary||'Pipeline paused for expert review.','warn','[HITL]');
        if(d.recommendation) log(`AI recommendation: ${d.recommendation.toUpperCase()}`,'info','[HITL]');
    });

    evtSource.addEventListener('result', (e) => {
        evtSource.close();
        const result = JSON.parse(e.data);
        nstate('validator','done');
        nstate('output','active');
        fstate('fa-val-out',true);
        setTimeout(()=>{ nstate('output','done'); fstate('fa-val-out',false); },400);
        renderRankings(result);
        renderTrace(result);
        orchState('complete');
        const elapsed=((Date.now()-t0)/1000).toFixed(1);
        document.getElementById('pipeMsg').innerHTML=`Completed in <em>${elapsed}s</em>`;
        running=false;
        document.getElementById('runBtn').disabled=false;
    });

    evtSource.addEventListener('error', (e) => {
        evtSource.close();
        try{
            const d = JSON.parse(e.data);
            log(d.message||'Pipeline error','warn','[ERR]');
        }catch(_){ log('Pipeline error','warn','[ERR]'); }
        orchState('idle'); stopTimer(); running=false;
        document.getElementById('runBtn').disabled=false;
    });

    evtSource.onerror = () => {
        if(!running) return;
        evtSource.close();
        log('Connection to server lost','warn','[ERR]');
        orchState('idle'); stopTimer(); running=false;
        document.getElementById('runBtn').disabled=false;
    };
}
```

**Step 4: Replace `hitl(action)` and remove `finalize()`**

Find and replace the entire `hitl` function (lines 807–827):
```javascript
function hitl(action){
    document.getElementById('hitlBanner').classList.remove('show');
    const msgs={
        approve:['Expert approved. Proceeding to L6 output.','ok','[HITL]'],
        reject: ['Expert rejected. Pipeline halted.','warn','[HITL]'],
        modify: ['Expert modification requested. Re-routing...','warn','[HITL]'],
        skip:   ['Validation skipped. Proceeding to L6 output.','info','[HITL]']
    };
    const [m,t,k]=msgs[action];
    log(m,t,k);
    if(action==='approve'||action==='skip'){
        nstate('validator','done');
        finalize();
    } else {
        nstate('validator','idle');
        orchState('idle');
        running=false;
        document.getElementById('runBtn').disabled=false;
        document.getElementById('pipeMsg').textContent='Pipeline rejected by expert.';
    }
}
```

Replace with:
```javascript
function hitl(action){
    document.getElementById('hitlBanner').classList.remove('show');
    const msgs={
        approve:['Expert approved. Resuming pipeline...','ok','[HITL]'],
        reject: ['Expert rejected. Pipeline will halt.','warn','[HITL]'],
        modify: ['Modification requested. Resuming...','warn','[HITL]'],
        skip:   ['Validation skipped. Resuming pipeline...','info','[HITL]']
    };
    const [m,t,k]=msgs[action]||['Decision sent.','info','[HITL]'];
    log(m,t,k);
    startTimer();
    orchState('running');
    fetch('/hitl',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({decision:action, note:''})
    });
}
```

Then find and delete the entire `finalize()` function (lines 829–846):
```javascript
async function finalize(){
    fstate('fa-val-out',true);
    nstate('output','active');
    log('Formatting output dict...','hot','[L6]');
    await wait(450);
    fstate('fa-val-out',false);
    nstate('output','done');
    log('Output: risk_rankings, risk_profile, traceability, skill_trace','ok','[L6]');
    orchState('complete');
    const e=((Date.now()-t0)/1000).toFixed(1);
    const m=Math.floor(e/60).toString().padStart(2,'0');
    const s=(e%60).toFixed(1).padStart(4,'0');
    document.getElementById('elapsedVal').textContent=`${m}:${s}`;
    document.getElementById('pipeMsg').innerHTML=`Completed in <em>${e}s</em>`;
    renderTrace();
    running=false;
    document.getElementById('runBtn').disabled=false;
}
```

**Step 5: Replace `renderRankings()` and `renderTrace()`**

Find and replace the entire `renderRankings` function (lines 851–872):
```javascript
function renderRankings(){
    const sorted=Object.entries(RESULTS).sort((a,b)=>b[1].score-a[1].score);
    ...
}
```

Replace with:
```javascript
function renderRankings(result){
    const rankings=(result.risk_rankings||[]);
    const el=document.getElementById('rankScroll');
    el.innerHTML='';
    // Update radar from real scores
    const radarScores={};
    rankings.forEach(item=>{ radarScores[item.domain_id]=item.risk_score; });
    updateRadar(radarScores);
    // Render each row
    rankings.forEach((item,i)=>{
        const sc=item.risk_score>.65?'hi':item.risk_score>.45?'med':'lo';
        const label=item.risk_score>.65?'HIGH':item.risk_score>.45?'MEDIUM':'LOW';
        const bel=item.high_belief||0;
        const div=document.createElement('div');
        div.className='ritem';
        div.innerHTML=`
            <div class="ritem-head">
                <span class="rdom">${item.domain_name}</span>
                <span class="rscore ${sc}">${(item.risk_score*100).toFixed(0)}%</span>
            </div>
            <div class="rbar-bg"><div class="rbar-fill ${sc}" style="width:0" data-w="${(item.risk_score*100).toFixed(0)}%"></div></div>
            <div class="rbel">Bel·High: ${(bel*100).toFixed(0)}% · ${label}</div>`;
        el.appendChild(div);
        setTimeout(()=>{
            div.classList.add('in');
            div.querySelector('.rbar-fill').style.width=div.querySelector('.rbar-fill').dataset.w;
        },i*110+40);
    });
}
```

Find and replace the entire `renderTrace` function (lines 874–879):
```javascript
function renderTrace(){
    document.getElementById('traceOut').innerHTML=`
        <div class="th">SI: 3 rules fired (corrosion + storm)</div>
        ...`;
}
```

Replace with:
```javascript
function renderTrace(result){
    const skillResults=(result.traceability||{}).skill_results||[];
    const skillTrace=result.skill_trace||{};
    const lines=skillResults.map(sr=>`<div class="th">${sr.domain_id}: ${(sr.activated_templates||[]).length} rules fired · score=${sr.risk_score.toFixed(2)}</div>`).join('');
    const routing=skillTrace.routing_decision||{};
    const validation=(skillTrace.expert_validation||'not_validated').toUpperCase();
    document.getElementById('traceOut').innerHTML=
        lines+`<div>Expert: ${validation}${skillTrace.expert_note?' — '+skillTrace.expert_note:''}</div>`;
}
```

**Step 6: Delete the `wait` helper (no longer needed)**

Find and remove:
```javascript
const wait=ms=>new Promise(r=>setTimeout(r,ms));
```

**Step 7: Manual integration test**

```bash
python server.py
```

Open browser at `http://localhost:5000`. Type a query such as:
```
assess CPP blade corrosion risk during storm conditions
```
Click RUN. Verify:
- [ ] Log entries stream in real-time as the pipeline runs
- [ ] Pipeline node states animate (orchestrator → rag → router → skill nodes → synthesis → validator)
- [ ] HITL banner appears when validator checkpoint is reached
- [ ] Clicking APPROVE/SKIP resumes the pipeline and shows output
- [ ] Rankings and trace populate with real data after completion
- [ ] Running a second query works correctly (state resets)

**Step 8: Commit**

```bash
git add dashboard.html
git commit -m "feat: wire dashboard to live pipeline via SSE — replaces static simulation"
```

---

### Task 7: Final regression check

**Step 1: Run full unit test suite**

```bash
pytest tests/ -m "not integration" -v
```
Expected: All tests pass.

**Step 2: Verify terminal mode still works**

```bash
python main.py "assess thruster bearing vibration risk"
```
Expected: Pipeline runs with terminal HITL prompt as before (not affected by web changes).

**Step 3: Commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: regression fixes from final test pass"
```
