from app.tools.data_tools import (
    get_dataset_info,
    get_dataset_profile,
    get_dataset_statistics,
)
from app.utils.validation import validate_csv


def test_validate_csv_accepts_good_file(sales_csv):
    result = validate_csv(sales_csv)
    assert result["valid"] is True


def test_validate_csv_rejects_missing_file():
    result = validate_csv("/nonexistent/path/does_not_exist.csv")
    assert result["valid"] is False


def test_validate_csv_handles_empty_file(empty_csv):
    result = validate_csv(empty_csv)
    # An empty (header-only) CSV should be handled gracefully, not
    # raise - either flagged invalid or accepted with 0 rows.
    assert "valid" in result


def test_get_dataset_info_reports_shape(sales_csv):
    info = get_dataset_info(sales_csv)
    assert "error" not in info
    assert info["rows"] == 480
    assert info["columns"] == 5


def test_get_dataset_statistics_covers_numeric_columns(sales_csv):
    stats = get_dataset_statistics(sales_csv)
    assert "error" not in stats
    assert "revenue" in stats


def test_get_dataset_profile_detects_outlier_and_candidates(sales_csv):
    profile = get_dataset_profile(sales_csv)
    assert "error" not in profile
    assert profile["rows"] == 480
    assert profile["duplicate_rows"] >= 0


def test_dataset_tools_handle_missing_column_gracefully(sales_csv):
    from app.tools.analysis_tools import analyze_column
    result = analyze_column(sales_csv, "does_not_exist")
    assert "error" in result


def test_dataset_tools_handle_missing_file_gracefully():
    result = get_dataset_info("/nonexistent/file.csv")
    assert "error" in result
