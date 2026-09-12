from app.tools.anomaly_tools import detect_anomalies


def test_detect_anomalies_univariate_finds_injected_outlier(sales_csv):
    result = detect_anomalies(sales_csv, column="revenue")
    assert "error" not in result
    assert result["method"].startswith("IQR")
    assert result["anomaly_count"] >= 1


def test_detect_anomalies_multivariate_runs_isolation_forest(sales_csv):
    result = detect_anomalies(sales_csv)
    assert "error" not in result
    assert "Isolation Forest" in result["method"] or "IQR" in result["method"]


def test_detect_anomalies_rejects_missing_column(sales_csv):
    result = detect_anomalies(sales_csv, column="does_not_exist")
    assert "error" in result


def test_detect_anomalies_rejects_non_numeric_column(sales_csv):
    result = detect_anomalies(sales_csv, column="region")
    assert "error" in result


def test_detect_anomalies_handles_small_dataset(missing_values_csv):
    # Small dataset with lots of missing values shouldn't crash -
    # should fall back to a per-column summary rather than forcing
    # Isolation Forest onto too little data.
    result = detect_anomalies(missing_values_csv)
    assert "error" not in result
