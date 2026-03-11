# Design: Live Query Input for Dashboard

**Date:** 2026-03-11
**Status:** Approved

## Problem

`dashboard.html` is a pure static simulation. The query textarea exists visually but is never read. `runPipeline()` replays hardcoded data with `setTimeout` delays and never calls the Python backend. Users cannot submit real queries.

## Solution

Connect the dashboard to the Python pipeline via a lightweight Flask server using Server-Sent Events (SSE) for real-time streaming and a `threading.Event` for HITL synchronisation.

## Architecture

```
Browser (dashboard.html)
  │
  ├─ POST /run  ─────────────────► server.py (Flask)
  │                                     │
  ├─ GET  /stream (EventSource) ◄───────┤  stdout → Queue → SSE
  │                                     │
  └─ POST /hitl  ────────────────►  threading.Event → validator resumes
```

## Components

### `server.py` (new)

- `POST /run` — accepts `{query, run_feedback}`, starts `run_pipeline()` in a `threading.Thread`, returns `200` immediately
- `GET /stream` — SSE endpoint; reads from a `queue.Queue` and yields `data:` lines; sends `event: result` with the final output dict when the pipeline completes
- `POST /hitl` — accepts `{decision, note}`, sets a `threading.Event` and stores the decision in a shared variable so `validator_agent.py` can read it
- Serves `dashboard.html` on `GET /`
- One pipeline runs at a time; a second `POST /run` while running returns `409`

### `agents/validator_agent.py` (modified)

- Import `core.hitl_bridge` (see below)
- Replace the `input()` block with `hitl_bridge.wait_for_decision()` which blocks on a `threading.Event`
- Read `expert_input` from `hitl_bridge.get_decision()` instead of terminal

### `core/hitl_bridge.py` (new)

Thin module exposing:
```python
def wait_for_decision() -> str   # blocks until POST /hitl fires the Event
def set_decision(decision: str, note: str)  # called by POST /hitl handler
def reset()                      # called at pipeline start
```

### `core/stream_queue.py` (new)

Thin module: a `queue.Queue` instance + a `QueueWriter` that wraps `sys.stdout` so all `print()` calls from every agent are captured without modifying agent files.

### `dashboard.html` (modified)

- `runPipeline()` replaced with real implementation:
  - `fetch('/run', {method:'POST', body: JSON.stringify({query, run_feedback})})`
  - `new EventSource('/stream')` — routes `message` events to `log()`, `event: hitl` to show HITL banner, `event: result` to render rankings/trace
- `hitl(action)` replaced with `fetch('/post /hitl', ...)` then continues listening to stream
- All hardcoded `RESULTS` data removed; rankings/trace rendered from real SSE `result` payload

### `requirements.txt` (modified)

Add `flask`.

## Data Flow

### SSE Event Types

| Event type | Payload | Dashboard action |
|------------|---------|-----------------|
| `message` (default) | `{tag, type, msg}` | Append to log |
| `hitl` | `{summary, flags, recommendation, ranking}` | Show HITL banner |
| `result` | Full output dict | Render rankings, belief intervals, trace |
| `error` | `{message}` | Show error state |

## Error Handling

- Pipeline exception → server sends `event: error`, resets shared state, re-enables RUN button
- `/run` called while running → `409 Conflict`, dashboard shows "Already running" toast
- `/stream` timeout (client disconnects) → server thread continues; queue is drained on reconnect or abandoned after pipeline completes

## Files Changed

| File | Type |
|------|------|
| `server.py` | New |
| `core/hitl_bridge.py` | New |
| `core/stream_queue.py` | New |
| `agents/validator_agent.py` | Modified — replace `input()` |
| `dashboard.html` | Modified — replace JS simulation |
| `requirements.txt` | Modified — add `flask` |
