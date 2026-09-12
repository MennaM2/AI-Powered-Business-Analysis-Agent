import pytest

from app.tools.sql_tools import run_sql_query, validate_sql_query


@pytest.mark.parametrize("query", [
    "SELECT * FROM dataset",
    "SELECT region, SUM(revenue) AS total FROM dataset GROUP BY region",
    "WITH t AS (SELECT * FROM dataset) SELECT * FROM t",
    "select * from dataset;",
])
def test_validate_sql_query_accepts_read_only_selects(query):
    result = validate_sql_query(query)
    assert result["valid"] is True


@pytest.mark.parametrize("query", [
    "DROP TABLE dataset",
    "DELETE FROM dataset",
    "UPDATE dataset SET revenue = 0",
    "INSERT INTO dataset VALUES (1)",
    "ALTER TABLE dataset ADD COLUMN x INT",
    "SELECT * FROM dataset; DROP TABLE dataset",
    "ATTACH DATABASE '/etc/passwd' AS x",
    "",
    "   ",
])
def test_validate_sql_query_rejects_unsafe_queries(query):
    result = validate_sql_query(query)
    assert result["valid"] is False
    assert "error" in result


def test_run_sql_query_executes_valid_query(sales_csv):
    result = run_sql_query(
        {"dataset": sales_csv},
        "SELECT region, SUM(revenue) AS total FROM dataset "
        "GROUP BY region ORDER BY total DESC",
    )
    assert result["status"] == "success"
    assert result["row_count"] == 4
    assert "region" in result["columns"]


def test_run_sql_query_blocks_destructive_query(sales_csv):
    result = run_sql_query({"dataset": sales_csv}, "DROP TABLE dataset")
    assert "error" in result


def test_run_sql_query_reports_bad_syntax(sales_csv):
    result = run_sql_query({"dataset": sales_csv}, "SELECT FROM WHERE garbage")
    assert "error" in result


def test_run_sql_query_truncates_large_results(sales_csv):
    result = run_sql_query({"dataset": sales_csv}, "SELECT * FROM dataset")
    assert result["status"] == "success"
    assert result["returned_rows"] <= 200


def test_run_sql_query_rejects_empty_datasets():
    result = run_sql_query({}, "SELECT * FROM dataset")
    assert "error" in result


def test_run_sql_query_joins_across_multiple_datasets(tmp_path):
    import pandas as pd

    orders_path = tmp_path / "orders.csv"
    users_path = tmp_path / "users.csv"
    pd.DataFrame({
        "order_id": [1, 2, 3],
        "user_id": [10, 20, 10],
    }).to_csv(orders_path, index=False)
    pd.DataFrame({
        "id": [10, 20],
        "country": ["Egypt", "USA"],
    }).to_csv(users_path, index=False)

    result = run_sql_query(
        {"orders": str(orders_path), "users": str(users_path)},
        "SELECT u.country, COUNT(*) AS cnt FROM orders o "
        "JOIN users u ON o.user_id = u.id GROUP BY u.country "
        "ORDER BY cnt DESC",
    )
    assert result["status"] == "success"
    assert result["tables_available"] == ["orders", "users"]
    assert result["rows"][0]["country"] == "Egypt"
    assert result["rows"][0]["cnt"] == 2


def test_run_sql_query_handles_string_columns_in_joins(tmp_path):
    """
    Regression test for a real production bug: duckdb==1.1.3 (as
    originally pinned) cannot register a pandas DataFrame that uses
    pandas 3.x's native 'str' extension dtype (the new default for
    text columns as of pandas 2.something/3.x), and fails with
    'Not implemented Error: Data type str not recognized' on any
    query touching a string column - including simple JOINs. This
    was reproduced exactly against real user data. Confirms the
    currently pinned duckdb version handles it correctly.
    """
    import pandas as pd

    orders_path = tmp_path / "orders.csv"
    users_path = tmp_path / "users.csv"
    pd.DataFrame({
        "order_id": [1, 2, 3],
        "user_id": [10, 20, 10],
        "status": ["Shipped", "Cancelled", "Shipped"],
    }).to_csv(orders_path, index=False)
    pd.DataFrame({
        "id": [10, 20],
        "country": ["Egypt", "USA"],
        "name": ["Alice", "Bob"],
    }).to_csv(users_path, index=False)

    # Confirm the fixture actually reproduces pandas' native 'str'
    # dtype for text columns - otherwise this test wouldn't be
    # testing anything.
    orders_df = pd.read_csv(orders_path)
    assert str(orders_df["status"].dtype) == "str"

    result = run_sql_query(
        {"orders": str(orders_path), "users": str(users_path)},
        "SELECT u.country, COUNT(*) AS cnt FROM orders o "
        "JOIN users u ON o.user_id = u.id "
        "WHERE o.status = 'Shipped' GROUP BY u.country",
    )
    assert "error" not in result
    assert result["status"] == "success"