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
