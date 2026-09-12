from app.tools.trend_tools import analyze_trends, compare_periods


def test_analyze_trends_detects_march_slump(sales_csv):
    result = analyze_trends(sales_csv, "date", "revenue", freq="month")
    assert "error" not in result
    assert result["overall_direction"] in ("increasing", "decreasing", "flat")
    assert len(result["periods"]) >= 2


def test_analyze_trends_rejects_missing_columns(sales_csv):
    result = analyze_trends(sales_csv, "not_a_column", "revenue")
    assert "error" in result


def test_analyze_trends_rejects_non_numeric_value_column(sales_csv):
    result = analyze_trends(sales_csv, "date", "region")
    assert "error" in result


def test_compare_periods_auto_selects_latest_two(sales_csv):
    result = compare_periods(sales_csv, "date", "revenue", freq="month")
    assert "error" not in result
    assert "percent_change" in result


def test_compare_periods_with_group_breakdown(sales_csv):
    result = compare_periods(
        sales_csv, "date", "revenue", freq="month", group_column="region"
    )
    assert "error" not in result
    assert "top_growth" in result
    assert "top_decline" in result
    assert len(result["top_growth"]) > 0


def test_compare_periods_rejects_unparseable_explicit_periods(sales_csv):
    result = compare_periods(
        sales_csv, "date", "revenue",
        freq="month", period_a="not-a-period", period_b="also-not"
    )
    assert "error" in result
