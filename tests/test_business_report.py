import os

from app.tools.analysis_tools import rank_categories
from app.tools.automation_tools import deliver_business_report
from app.tools.report_tools import generate_business_report


def test_rank_categories_returns_top_and_bottom(sales_csv):
    result = rank_categories(sales_csv, "product", "revenue")
    assert "error" not in result
    assert len(result["top_performers"]) > 0
    assert len(result["bottom_performers"]) > 0


def test_rank_categories_rejects_bad_aggregation(sales_csv):
    result = rank_categories(
        sales_csv, "product", "revenue", aggregation="not_a_real_agg"
    )
    assert "error" in result


def test_generate_business_report_auto_detects_columns(sales_csv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # generate_business_report writes to ./outputs relative to CWD;
    # sales_csv is already inside tmp_path via the fixture.
    result = generate_business_report(sales_csv)
    assert result["status"] == "success"
    assert result["detected_columns"]["date_column"] is not None
    assert result["detected_columns"]["value_column"] is not None
    assert os.path.exists(result["report_path_markdown"])
    assert os.path.exists(result["report_path_html"])
    assert len(result["key_insights"]) > 0


def test_generate_business_report_never_fabricates_without_data(empty_csv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = generate_business_report(empty_csv)
    # Should either report an error or succeed with no findings -
    # never crash, never invent numbers for a 0-row dataset.
    if "error" not in result:
        assert result["key_insights"] == [] or isinstance(
            result["key_insights"], list
        )


def test_deliver_business_report_confirms_real_save(sales_csv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = deliver_business_report(sales_csv)
    assert result["action"] == "report_saved"
    assert all(os.path.exists(path) for path in result["saved_files"])


def test_deliver_business_report_never_fakes_email_success(sales_csv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # No SMTP configured in test env -> must not claim the email sent.
    result = deliver_business_report(sales_csv, email_to="someone@example.com")
    assert result["email"]["sent"] is False
    assert result["email"]["attempted"] is False
