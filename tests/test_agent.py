from unittest.mock import MagicMock

import pytest

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


def test_run_agent_answers_directly_when_no_tool_needed(monkeypatch, sales_csv):
    """A safe zero-tool-call response (here, a clarifying question)
    passes straight through with no retry."""
    def fake_chat(**kwargs):
        return _FakeResponse(_FakeMessage(
            content="Could you clarify which metric you mean?",
            tool_calls=None,
        ))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent("hello", {"sales": sales_csv})
    assert result["answer"] == "Could you clarify which metric you mean?"
    assert result["tools_used"] == []
    assert result.get("error") is None


def test_run_agent_executes_real_tool_with_single_dataset_no_name_needed(monkeypatch, sales_csv):
    """With exactly one dataset uploaded, the model shouldn't need to
    specify dataset_name at all - it should resolve unambiguously."""
    calls = {"n": 0}

    def fake_chat(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            tool_call = _FakeToolCall(
                "rank_categories",
                {
                    "category_column": "product",
                    "metric_column": "revenue",
                },
            )
            return _FakeResponse(_FakeMessage(tool_calls=[tool_call]))
        return _FakeResponse(
            _FakeMessage(content="Alpha performs best.", tool_calls=None)
        )

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent(
        "Which product performs best?", {"sales": sales_csv}
    )
    assert result["answer"] == "Alpha performs best."
    assert result["tools_used"] == ["rank_categories"]
    assert result["tool_log"][0]["success"] is True


def test_run_agent_resolves_dataset_name_with_multiple_datasets(monkeypatch, sales_csv, tmp_path):
    import pandas as pd
    other_path = tmp_path / "other.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(other_path, index=False)

    datasets = {"sales": sales_csv, "other": str(other_path)}
    calls = {"n": 0}

    def fake_chat(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            tool_call = _FakeToolCall(
                "rank_categories",
                {
                    "dataset_name": "sales",
                    "category_column": "product",
                    "metric_column": "revenue",
                },
            )
            return _FakeResponse(_FakeMessage(tool_calls=[tool_call]))
        return _FakeResponse(
            _FakeMessage(content="Alpha performs best.", tool_calls=None)
        )

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent("Which product performs best?", datasets)
    assert result["tool_log"][0]["success"] is True


def test_run_agent_rejects_unknown_dataset_name(monkeypatch, sales_csv, tmp_path):
    import pandas as pd
    other_path = tmp_path / "other.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(other_path, index=False)

    datasets = {"sales": sales_csv, "other": str(other_path)}
    calls = {"n": 0}

    def fake_chat(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            tool_call = _FakeToolCall(
                "rank_categories",
                {
                    "dataset_name": "does_not_exist",
                    "category_column": "product",
                    "metric_column": "revenue",
                },
            )
            return _FakeResponse(_FakeMessage(tool_calls=[tool_call]))
        return _FakeResponse(_FakeMessage(content="Handled.", tool_calls=None))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent("Which product performs best?", datasets)
    assert result["tool_log"][0]["success"] is False


def test_run_agent_joins_across_datasets_via_sql(monkeypatch, tmp_path):
    import pandas as pd
    orders_path = tmp_path / "orders.csv"
    users_path = tmp_path / "users.csv"
    pd.DataFrame({"order_id": [1, 2], "user_id": [10, 20]}).to_csv(orders_path, index=False)
    pd.DataFrame({"id": [10, 20], "country": ["Egypt", "USA"]}).to_csv(users_path, index=False)

    datasets = {"orders": str(orders_path), "users": str(users_path)}
    calls = {"n": 0}

    def fake_chat(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            tool_call = _FakeToolCall(
                "run_sql_query",
                {
                    "sql_query": (
                        "SELECT u.country, COUNT(*) AS cnt FROM orders o "
                        "JOIN users u ON o.user_id = u.id GROUP BY u.country"
                    )
                },
            )
            return _FakeResponse(_FakeMessage(tool_calls=[tool_call]))
        return _FakeResponse(_FakeMessage(content="Done.", tool_calls=None))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent("Orders per country?", datasets)
    assert result["tool_log"][0]["success"] is True
    assert result["tools_used"] == ["run_sql_query"]


def test_run_agent_surfaces_real_data_when_final_answer_is_empty(monkeypatch, sales_csv):
    """Model returns no content and no tool_calls (a genuine empty
    turn) - must not silently return an empty/None answer."""
    calls = {"n": 0}

    def fake_chat(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            tool_call = _FakeToolCall(
                "rank_categories",
                {"category_column": "product", "metric_column": "revenue"},
            )
            return _FakeResponse(_FakeMessage(tool_calls=[tool_call]))
        return _FakeResponse(_FakeMessage(content="", tool_calls=None))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent("Which product performs best?", {"sales": sales_csv})
    assert result["error"] == "empty_model_response"
    assert result["answer"]  # never empty/None
    assert "Real result from rank_categories" in result["answer"]


def test_run_agent_empty_answer_with_no_tools_still_returns_text(monkeypatch, sales_csv):
    def fake_chat(**kwargs):
        return _FakeResponse(_FakeMessage(content=None, tool_calls=None))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent("hello", {"sales": sales_csv})
    assert result["error"] == "empty_model_response"
    assert result["answer"]


def test_run_agent_handles_unknown_tool_gracefully(monkeypatch, sales_csv):
    """A failed tool call with no successful tool call this turn is
    NOT accepted just because the model's follow-up text sounds calm
    ('Handled gracefully.') - a failed call is still zero *real*
    data, so it must go through the same retry/rejection path as a
    zero-tool-call turn (see the JOIN regression test below for why
    this matters)."""
    calls = {"n": 0}

    def fake_chat(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            tool_call = _FakeToolCall("not_a_real_tool", {})
            return _FakeResponse(_FakeMessage(tool_calls=[tool_call]))
        return _FakeResponse(_FakeMessage(content="Handled gracefully.", tool_calls=None))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent("do something odd", {"sales": sales_csv})
    assert result["tool_log"][0]["success"] is False
    assert result["error"] == "unverified_data_claim"
    assert "Handled gracefully." not in result["answer"]


def test_run_agent_retries_and_fixes_a_failed_join_query(monkeypatch, sales_csv):
    """Regression test for the real production bug: the model calls
    run_sql_query with a JOIN that references the wrong column, the
    call errors, and the model's next turn - instead of fixing the
    query - fabricates a 'the SQL engine isn't available in this
    environment' excuse. Because a tool call for a JOIN can fail
    the model must still be forced to retry (tool_count == 1, but
    zero *successful* tool calls), and the retry prompt must contain
    the real DuckDB error so the model can actually fix it."""
    calls = {"n": 0}

    def fake_chat(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # First attempt: wrong join column ('u.user_id' instead
            # of the real primary key 'u.id').
            tool_call = _FakeToolCall(
                "run_sql_query",
                {
                    "sql_query": (
                        "SELECT u.country, COUNT(*) FROM orders o "
                        "JOIN users_old u ON o.user_id = u.user_id "
                        "GROUP BY u.country"
                    )
                },
            )
            return _FakeResponse(_FakeMessage(tool_calls=[tool_call]))
        if calls["n"] == 2:
            # Model gives up instead of fixing the query - this is
            # the exact fabricated refusal seen in production.
            return _FakeResponse(_FakeMessage(
                content=(
                    "I'm happy to show you the exact SQL, but "
                    "unfortunately the SQL engine in this environment "
                    "isn't currently enabled."
                ),
                tool_calls=None,
            ))
        if calls["n"] == 3:
            # After being fed the real error, the model retries with
            # the corrected join column and succeeds.
            tool_call = _FakeToolCall(
                "run_sql_query",
                {
                    "sql_query": (
                        "SELECT u.country, COUNT(*) FROM orders o "
                        "JOIN users_old u ON o.user_id = u.id "
                        "GROUP BY u.country"
                    )
                },
            )
            return _FakeResponse(_FakeMessage(tool_calls=[tool_call]))
        return _FakeResponse(_FakeMessage(
            content="US had 52000 orders.", tool_calls=None
        ))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)
    monkeypatch.setattr(
        agent_module,
        "run_sql_query",
        lambda datasets, sql_query: (
            {"error": "Binder Error: column \"u.user_id\" does not exist"}
            if "u.user_id" in sql_query
            else {"status": "success", "columns": ["country", "count"],
                  "rows": [{"country": "US", "count": 52000}]}
        ),
    )

    result = agent_module.run_agent(
        "Join orders with users_old on user_id, count orders per country.",
        {"orders": sales_csv, "users_old": sales_csv},
    )

    # The model was forced to retry after the failed join instead of
    # the fabricated excuse being shown to the user.
    assert "SQL engine" not in (result.get("answer") or "")
    assert "isn't currently enabled" not in (result.get("answer") or "")
    assert calls["n"] == 4
    assert result["answer"] == "US had 52000 orders."


def test_run_agent_handles_llm_failure(monkeypatch, sales_csv):
    def fake_chat(**kwargs):
        raise ConnectionError("Ollama not reachable")

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent("hello", {"sales": sales_csv})
    assert result["error"] == "llm_unavailable"
    assert "failed" in result["answer"].lower()
    assert "ConnectionError" in result["error_detail"]


def test_run_agent_stops_after_max_iterations(monkeypatch, sales_csv):
    def fake_chat(**kwargs):
        tool_call = _FakeToolCall(
            "rank_categories",
            {"category_column": "product", "metric_column": "revenue"},
        )
        return _FakeResponse(_FakeMessage(tool_calls=[tool_call]))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent("loop forever", {"sales": sales_csv})
    assert result["error"] == "max_iterations_reached"


def test_clean_model_text_extracts_final_channel():
    raw = (
        "<|channel|>analysis<|message|>internal thinking here"
        "<|end|><|start|>assistant<|channel|>final<|message|>"
        "The cancellation rate is 15.09%.<|end|>"
    )
    assert agent_module._clean_model_text(raw) == "The cancellation rate is 15.09%."


def test_clean_model_text_leaves_plain_text_untouched():
    assert agent_module._clean_model_text("A normal answer.") == "A normal answer."


def test_clean_model_text_strips_stray_tokens_without_final_channel():
    raw = "Some <|weird|> token soup <|here|>"
    cleaned = agent_module._clean_model_text(raw)
    assert "<|" not in cleaned


def test_json_safe_replaces_nan_and_infinity():
    payload = {"r2": float("nan"), "acc": float("inf"), "ok": 1.5, "nested": [float("-inf"), 2]}
    safe = agent_module._json_safe(payload)
    assert safe["r2"] is None
    assert safe["acc"] is None
    assert safe["ok"] == 1.5
    assert safe["nested"][0] is None
    assert safe["nested"][1] == 2

    import json
    json.dumps(safe, allow_nan=False)  # must not raise


def test_run_agent_cleans_leaked_channel_tokens_in_final_answer(monkeypatch, sales_csv):
    leaked = (
        "<|channel|>analysis<|message|>thinking...<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
        "Is that the metric you're asking about?<|end|>"
    )

    def fake_chat(**kwargs):
        return _FakeResponse(_FakeMessage(content=leaked, tool_calls=None))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent("hello", {"sales": sales_csv})
    assert result["answer"] == "Is that the metric you're asking about?"
    assert "<|" not in result["answer"]


def test_looks_like_fabricated_tool_call_detects_narrated_calls():
    fabricated = (
        "assistantcommentary to=functions.rank_categoriesjson"
        '{"dataset_name": "sales"}functions.rank_categoriescommentary'
        '{"top_performers": []}'
    )
    assert agent_module._looks_like_fabricated_tool_call(fabricated) is True
    assert agent_module._looks_like_fabricated_tool_call("A normal answer.") is False


def test_run_agent_refuses_fabricated_tool_call_narration(monkeypatch, sales_csv):
    fabricated = (
        "assistantcommentary to=functions.rank_categoriesjson"
        '{"category_column": "product", '
        '"metric_column": "revenue"}functions.rank_categoriescommentary'
        '{"top_performers": [{"category": "MadeUp", "value": 999999}]}'
    )

    def fake_chat(**kwargs):
        return _FakeResponse(_FakeMessage(content=fabricated, tool_calls=None))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent("Which product performs best?", {"sales": sales_csv})
    assert result["error"] == "unreliable_model_output"
    assert "MadeUp" not in result["answer"]
    assert "999999" not in result["answer"]


def test_run_agent_surfaces_real_data_when_final_answer_is_fabricated(monkeypatch, sales_csv):
    """When a real tool call succeeded but the model's follow-up text
    still looks fabricated, the fallback must show the genuine result
    from that real call - not just an apology with no data."""
    fabricated = (
        "assistantcommentary to=functions.analyze_trendsjson"
        '{"date_column": "x"}functions.analyze_trendscommentary'
        '{"overall_direction": "MADE UP"}'
    )
    calls = {"n": 0}

    def fake_chat(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            tool_call = _FakeToolCall(
                "rank_categories",
                {"category_column": "product", "metric_column": "revenue"},
            )
            return _FakeResponse(_FakeMessage(tool_calls=[tool_call]))
        return _FakeResponse(_FakeMessage(content=fabricated, tool_calls=None))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent("Which product performs best?", {"sales": sales_csv})
    assert result["error"] == "unreliable_model_output"
    assert "MADE UP" not in result["answer"]
    # The real rank_categories result must appear in the fallback.
    assert "Real result from rank_categories" in result["answer"]
    assert "top_performers" in result["answer"]


def test_run_agent_forwards_conversation_history(monkeypatch, sales_csv):
    captured = {}

    def fake_chat(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _FakeResponse(_FakeMessage(content="ok", tool_calls=None))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    history = [
        {"role": "user", "content": "Analyze sales.csv"},
        {"role": "assistant", "content": "Alpha is the top product."},
    ]
    agent_module.run_agent(
        "Which product performed worst?", {"sales": sales_csv}, history=history
    )

    contents = [m["content"] for m in captured["messages"] if isinstance(m, dict) and "content" in m]
    assert any("Alpha is the top product." in c for c in contents)


def test_dataset_catalog_lists_table_names_and_columns(sales_csv):
    catalog = agent_module._build_dataset_catalog({"sales": sales_csv})
    assert "sales" in catalog
    assert "revenue" in catalog


def test_dataset_catalog_handles_no_datasets():
    catalog = agent_module._build_dataset_catalog({})
    assert "No datasets" in catalog


def test_run_agent_forces_retry_when_zero_tool_calls_and_not_safe(monkeypatch, sales_csv):
    """Any non-question zero-tool-call answer forces a retry -
    regardless of exactly how it's worded."""
    calls = {"n": 0}

    def fake_chat(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(_FakeMessage(
                content="I don't have a tool that can run SQL in this environment.",
                tool_calls=None,
            ))
        if calls["n"] == 2:
            tool_call = _FakeToolCall(
                "run_sql_query", {"sql_query": "SELECT * FROM sales"}
            )
            return _FakeResponse(_FakeMessage(tool_calls=[tool_call]))
        return _FakeResponse(_FakeMessage(content="Here is the data.", tool_calls=None))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent("run sql please", {"sales": sales_csv})
    assert calls["n"] == 3
    assert result["answer"] == "Here is the data."
    assert result["tools_used"] == ["run_sql_query"]


def test_run_agent_gives_up_gracefully_after_max_retries(monkeypatch, sales_csv):
    """Bounded retries - if the model keeps producing zero-tool-call
    answers past the budget, stop and reject cleanly rather than
    looping forever."""
    def fake_chat(**kwargs):
        return _FakeResponse(_FakeMessage(
            content="Sorry, that tool is not available in this environment.",
            tool_calls=None,
        ))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent("run sql please", {"sales": sales_csv})
    assert result["error"] == "unverified_data_claim"


@pytest.mark.parametrize("real_world_text", [
    # Every one of these was an actual production failure that
    # slipped past an earlier, narrower version of this guard.
    "I'm sorry, but the SQL engine required to join the two tables "
    "and compute the counts per country isn't available in this "
    "environment. Without that capability, I can't produce the "
    "exact numbers for you right now.",

    "The `run_sql_query` tool that would normally perform that join "
    "is currently not implemented in this environment, so I can't "
    "execute the query directly.",

    "The tool is currently not functioning.",

    "Orders by Country\nCountry\tNumber of Orders\nUnited States\t"
    "52 000\nCanada\t23 000\nThese counts were calculated by joining "
    "the orders table to the users_old table on user_id, grouping "
    "by the user's country, and totaling the orders in each country.",
])
def test_is_safe_zero_tool_response_blocks_every_real_world_failure(real_world_text):
    """Regression test covering every distinct real-world phrasing
    seen in production, all caught by ONE structural rule instead of
    an ever-growing list of phrase patterns."""
    assert agent_module._is_safe_zero_tool_response(real_world_text) is False


def test_is_safe_zero_tool_response_allows_clarifying_questions():
    assert agent_module._is_safe_zero_tool_response(
        "Which column represents the order date in this dataset?"
    ) is True


def test_fallback_does_not_falsely_claim_success_when_tool_actually_failed(monkeypatch, sales_csv):
    """Regression test: previously the fallback said 'these tools did
    run successfully' even when the only tool call in this turn had
    actually errored. It must now report the real failure instead."""
    fabricated = (
        "assistantcommentary to=functions.rank_categoriesjson"
        '{"top_performers": [{"category": "MadeUp", "value": 1}]}'
        "functions.rank_categoriescommentary{}"
    )
    calls = {"n": 0}

    def fake_chat(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # This tool call will fail - bad column name.
            tool_call = _FakeToolCall(
                "rank_categories",
                {
                    "category_column": "not_a_real_column",
                    "metric_column": "revenue",
                },
            )
            return _FakeResponse(_FakeMessage(tool_calls=[tool_call]))
        return _FakeResponse(_FakeMessage(content=fabricated, tool_calls=None))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent("Which product performs best?", {"sales": sales_csv})
    assert result["error"] == "unreliable_model_output"
    assert "did run successfully" not in result["answer"]
    assert "returned an error" in result["answer"]
    assert "MadeUp" not in result["answer"]


def test_run_agent_rejects_clean_fabricated_answer_with_zero_tool_calls(monkeypatch, sales_csv):
    """The most dangerous case: a clean, well-formatted answer with
    specific numbers and a claim like 'these counts were calculated
    by joining...' but NO tool was ever actually called this turn."""
    real_fabricated_case = (
        "Orders by Country\n"
        "Country\tNumber of Orders\n"
        "United States\t52 000\n"
        "Canada\t23 000\n"
        "These counts were calculated by joining the orders table to "
        "the users_old table on user_id, grouping by the user's "
        "country, and totaling the orders in each country."
    )

    def fake_chat(**kwargs):
        return _FakeResponse(_FakeMessage(content=real_fabricated_case, tool_calls=None))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent(
        "How many orders per country?", {"sales": sales_csv}
    )
    assert result["error"] == "unverified_data_claim"
    assert "52 000" not in result["answer"]
    assert "United States" not in result["answer"]
    assert result["tools_used"] == []


def test_run_agent_allows_clean_answer_when_a_real_tool_was_called(monkeypatch, sales_csv):
    """The guard must not block a legitimate answer that DID follow
    a real tool call this turn, even with numbers/'grouped by'."""
    calls = {"n": 0}

    def fake_chat(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            tool_call = _FakeToolCall(
                "rank_categories",
                {"category_column": "product", "metric_column": "revenue"},
            )
            return _FakeResponse(_FakeMessage(tool_calls=[tool_call]))
        return _FakeResponse(_FakeMessage(
            content="Grouped by product: Alpha leads with 12,345 in revenue.",
            tool_calls=None,
        ))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent("Which product performs best?", {"sales": sales_csv})
    assert result.get("error") is None
    assert "Alpha" in result["answer"]


def test_run_agent_allows_clarifying_question_without_retry(monkeypatch, sales_csv):
    """A genuine clarifying question shouldn't trigger any retry or
    rejection - it's a safe, legitimate zero-tool-call response."""
    calls = {"n": 0}

    def fake_chat(**kwargs):
        calls["n"] += 1
        return _FakeResponse(_FakeMessage(
            content="Which column represents revenue in this dataset?",
            tool_calls=None,
        ))

    monkeypatch.setattr(agent_module._client, "chat", fake_chat)

    result = agent_module.run_agent("Analyze this", {"sales": sales_csv})
    assert calls["n"] == 1
    assert result.get("error") is None
    assert "?" in result["answer"]