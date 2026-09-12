import os

# Route the persistent session store to a throwaway test DB, set
# before any test module can import app.config/app.agent.memory (this
# conftest.py is collected first by pytest), so tests never touch a
# real data/sessions.db.
os.environ.setdefault("SESSIONS_DB_PATH", "/tmp/test_agent_sessions.db")
if os.path.exists("/tmp/test_agent_sessions.db"):
    os.remove("/tmp/test_agent_sessions.db")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sales_csv(tmp_path):
    """A realistic small business dataset: dates, region, product,
    revenue, units - with a seasonal dip and one injected outlier so
    trend/anomaly/ranking tools all have something real to find."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    regions = ["North", "South", "East", "West"]
    products = ["Alpha", "Beta", "Gamma"]

    rows = []
    for date in dates:
        for _ in range(4):
            revenue = rng.normal(1000, 150)
            if date.month == 3:
                revenue *= 0.6  # simulate a March slump
            rows.append([
                date.strftime("%Y-%m-%d"),
                rng.choice(regions),
                rng.choice(products),
                max(float(revenue), 0.0),
                int(rng.integers(1, 20)),
            ])

    df = pd.DataFrame(
        rows, columns=["date", "region", "product", "revenue", "units"]
    )
    df.loc[5, "revenue"] = 50_000.0  # obvious outlier

    path = tmp_path / "sales.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def empty_csv(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("col_a,col_b\n")
    return str(path)


@pytest.fixture
def malformed_csv(tmp_path):
    path = tmp_path / "malformed.csv"
    path.write_text("this is not,,a,,,valid\ncsv\"file\n\n\n")
    return str(path)


@pytest.fixture
def missing_values_csv(tmp_path):
    df = pd.DataFrame({
        "region": ["North", "South", None, "East", "West"] * 4,
        "revenue": [100, None, 300, 400, None] * 4,
    })
    path = tmp_path / "missing.csv"
    df.to_csv(path, index=False)
    return str(path)