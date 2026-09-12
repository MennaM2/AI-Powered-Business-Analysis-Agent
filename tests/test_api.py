from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model" in body


def test_upload_rejects_non_csv():
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_accepts_valid_csv(sales_csv):
    with open(sales_csv, "rb") as file:
        response = client.post(
            "/upload", files={"file": ("sales.csv", file, "text/csv")}
        )
    assert response.status_code == 200
    body = response.json()
    assert "session_id" in body
    assert body["dataset"]["rows"] > 0
    assert body["dataset"]["table_name"] == "sales"
    assert body["all_datasets"] == ["sales"]


def test_second_upload_to_same_session_adds_not_replaces(sales_csv, tmp_path):
    import pandas as pd
    other_path = tmp_path / "regions.csv"
    pd.DataFrame({"region": ["North", "South"], "manager": ["A", "B"]}).to_csv(
        other_path, index=False
    )

    with open(sales_csv, "rb") as file:
        first = client.post(
            "/upload", files={"file": ("sales.csv", file, "text/csv")}
        )
    session_id = first.json()["session_id"]

    with open(other_path, "rb") as file:
        second = client.post(
            "/upload",
            files={"file": ("regions.csv", file, "text/csv")},
            params={"session_id": session_id},
        )

    assert second.status_code == 200
    body = second.json()
    assert body["session_id"] == session_id
    # Both datasets present - the first one was NOT wiped.
    assert set(body["all_datasets"]) == {"sales", "regions"}


def test_chat_requires_known_session():
    response = client.post(
        "/chat", json={"session_id": "not-a-real-session", "message": "hi"}
    )
    assert response.status_code == 404


def test_report_requires_file_path_or_session():
    response = client.post("/report", json={})
    assert response.status_code == 400


def test_list_sessions_includes_new_session(sales_csv):
    with open(sales_csv, "rb") as file:
        upload_response = client.post(
            "/upload", files={"file": ("sales.csv", file, "text/csv")}
        )
    session_id = upload_response.json()["session_id"]

    response = client.get("/sessions")
    assert response.status_code == 200
    session_ids = [s["session_id"] for s in response.json()]
    assert session_id in session_ids


def test_get_session_detail_returns_datasets_and_history(sales_csv, monkeypatch):
    import app.agent.agent as agent_module

    class _FakeToolCall:
        def __init__(self, name, arguments):
            self.function = MagicMock()
            self.function.name = name
            self.function.arguments = arguments

    class _FakeMessage:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class _FakeResponse:
        def __init__(self, message):
            self.message = message

    def fake_chat(**kwargs):
        return _FakeResponse(_FakeMessage(content="Hi there.", tool_calls=None))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    with open(sales_csv, "rb") as file:
        upload_response = client.post(
            "/upload", files={"file": ("sales.csv", file, "text/csv")}
        )
    session_id = upload_response.json()["session_id"]

    client.post("/chat", json={"session_id": session_id, "message": "hello"})

    detail_response = client.get(f"/sessions/{session_id}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert "sales" in detail["datasets"]
    assert len(detail["history"]) == 2  # user turn + assistant turn
    assert detail["history"][0]["role"] == "user"


def test_get_session_detail_404_for_unknown_session():
    response = client.get("/sessions/not-a-real-session-id")
    assert response.status_code == 404


def test_end_to_end_upload_then_report(sales_csv):
    """Upload -> generate report (deterministic path, no LLM needed)."""
    with open(sales_csv, "rb") as file:
        upload_response = client.post(
            "/upload", files={"file": ("sales.csv", file, "text/csv")}
        )
    assert upload_response.status_code == 200
    session_id = upload_response.json()["session_id"]

    report_response = client.post("/report", json={"session_id": session_id})
    assert report_response.status_code == 200
    body = report_response.json()
    assert body["status"] == "success"
    assert body["report_path_markdown"]
    assert len(body["key_insights"]) >= 0


def test_report_requires_dataset_name_when_session_has_multiple(sales_csv, tmp_path):
    import pandas as pd
    other_path = tmp_path / "extra.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(other_path, index=False)

    with open(sales_csv, "rb") as file:
        first = client.post(
            "/upload", files={"file": ("sales.csv", file, "text/csv")}
        )
    session_id = first.json()["session_id"]

    with open(other_path, "rb") as file:
        client.post(
            "/upload",
            files={"file": ("extra.csv", file, "text/csv")},
            params={"session_id": session_id},
        )

    # No dataset_name given, and session now has 2 datasets -> 400.
    response = client.post("/report", json={"session_id": session_id})
    assert response.status_code == 400

    # With dataset_name specified, it works.
    response = client.post(
        "/report", json={"session_id": session_id, "dataset_name": "sales"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_end_to_end_upload_then_chat_with_mocked_llm(sales_csv, monkeypatch):
    """Upload -> ask a business question -> agent selects a tool ->
    real tool executes -> final answer returned. The LLM itself is
    mocked so this test doesn't depend on Ollama being available."""
    import app.agent.agent as agent_module

    class _FakeToolCall:
        def __init__(self, name, arguments):
            self.function = MagicMock()
            self.function.name = name
            self.function.arguments = arguments

    class _FakeMessage:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls
            self.role = "assistant"

    class _FakeResponse:
        def __init__(self, message):
            self.message = message

    calls = {"n": 0}

    def fake_chat(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            tool_call = _FakeToolCall("get_dataset_profile", {})
            return _FakeResponse(_FakeMessage(tool_calls=[tool_call]))
        return _FakeResponse(
            _FakeMessage(content="This dataset has 480 rows.", tool_calls=None)
        )

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    with open(sales_csv, "rb") as file:
        upload_response = client.post(
            "/upload", files={"file": ("sales.csv", file, "text/csv")}
        )
    session_id = upload_response.json()["session_id"]

    chat_response = client.post(
        "/chat",
        json={"session_id": session_id, "message": "Tell me about this dataset."},
    )
    assert chat_response.status_code == 200
    body = chat_response.json()
    assert body["answer"] == "This dataset has 480 rows."
    assert "get_dataset_profile" in body["tools_used"]