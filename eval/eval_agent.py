"""
LLM evaluation harness for the business-analysis agent.

This is NOT a unit test with a mocked LLM (see tests/test_agent.py for
that). This runs each question through the REAL agent -> REAL Ollama
model -> REAL tools, exactly like a user would, then checks two
things a unit test can't:

1. Tool grounding: did the agent actually call one of the tools this
   question requires, instead of answering from the model's own
   (possibly wrong) assumptions about the data?
2. Factual correctness: does a number in the agent's final answer
   match a ground-truth value computed independently, directly from
   the CSV with pandas/duckdb - not from the agent itself?

A case can fail tool grounding but "sound" confident, or pass tool
grounding but still get the number wrong (bad reasoning over a
correct tool result) - this harness catches both failure modes
separately instead of collapsing them into one pass/fail.

Usage:
    python eval/eval_agent.py
    python eval/eval_agent.py --save eval/eval_results.json

Requires a reachable Ollama (or whatever AGENT_MODEL is configured to
use in app/config.py) - this is an integration eval, not a fast unit
test, and is not run as part of `pytest tests/`.
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.agent import run_agent  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ORDERS_CSV = str(REPO_ROOT / "data" / "uploads" / "orders.csv")
USERS_CSV = str(REPO_ROOT / "data" / "uploads" / "users_old.csv")


def _numbers_in(text: str) -> list[float]:
    """Pull every number out of free text, e.g. '124,503 orders' or
    '43.8%' -> [124503.0, 43.8]. Commas are stripped so formatted
    numbers still match."""
    cleaned = text.replace(",", "")
    return [float(match) for match in re.findall(r"-?\d+\.?\d*", cleaned)]


def _answer_contains_number(answer: str, expected: float, rel_tol: float = 0.02) -> bool:
    """True if some number in the answer is within rel_tol of
    `expected` (default 2%, since the model may round)."""
    if expected == 0:
        return any(abs(n) < 1e-6 for n in _numbers_in(answer))
    return any(
        abs(n - expected) / abs(expected) <= rel_tol
        for n in _numbers_in(answer)
    )


@dataclass
class EvalCase:
    id: str
    question: str
    datasets: dict
    expected_tools: list  # at least one of these must appear in tools_used
    check: Optional[Callable[[dict], tuple]] = None  # (result) -> (passed, detail)
    ground_truth_note: str = ""


@dataclass
class EvalResult:
    case_id: str
    question: str
    tool_grounded: bool
    tools_used: list
    factually_correct: Optional[bool]
    detail: str
    latency_seconds: float
    error: Optional[str]


def _load_ground_truth():
    """Compute expected values directly from the CSVs - independent
    of any agent/tool code, so a bug in a tool can't also make its
    own eval check pass."""
    orders = pd.read_csv(ORDERS_CSV)
    users = pd.read_csv(USERS_CSV)

    merged = orders.merge(
        users, left_on="user_id", right_on="id", suffixes=("", "_user")
    )
    top_country = merged["country"].value_counts().idxmax()
    top_country_count = int(merged["country"].value_counts().max())

    return {
        "row_count": len(orders),
        "duplicate_rows": int(orders.duplicated().sum()),
        "avg_num_of_item": float(orders["num_of_item"].mean()),
        "returned_at_missing_pct": float(orders["returned_at"].isna().mean() * 100),
        "top_status": orders["status"].value_counts().idxmax(),
        "top_status_count": int(orders["status"].value_counts().max()),
        "top_country": top_country,
        "top_country_count": top_country_count,
    }


GT = _load_ground_truth()


CASES = [
    EvalCase(
        id="row_count",
        question="How many rows are in this dataset?",
        datasets={"orders": ORDERS_CSV},
        expected_tools=["get_dataset_info", "get_dataset_profile", "get_dataset_statistics"],
        check=lambda r: (
            _answer_contains_number(r["answer"], GT["row_count"]),
            f"expected ~{GT['row_count']}",
        ),
        ground_truth_note="pandas len(df)",
    ),
    EvalCase(
        id="duplicate_rows",
        question="Are there any duplicate rows in this dataset? How many?",
        datasets={"orders": ORDERS_CSV},
        expected_tools=["get_dataset_info", "get_dataset_profile"],
        check=lambda r: (
            _answer_contains_number(r["answer"], GT["duplicate_rows"]),
            f"expected {GT['duplicate_rows']}",
        ),
        ground_truth_note="pandas df.duplicated().sum()",
    ),
    EvalCase(
        id="avg_num_of_item",
        question="What is the average number of items per order (num_of_item)?",
        datasets={"orders": ORDERS_CSV},
        expected_tools=["get_dataset_statistics", "analyze_column", "get_dataset_profile", "run_sql_query"],
        check=lambda r: (
            _answer_contains_number(r["answer"], GT["avg_num_of_item"], rel_tol=0.05),
            f"expected ~{GT['avg_num_of_item']:.2f}",
        ),
        ground_truth_note="pandas df['num_of_item'].mean()",
    ),
    EvalCase(
        id="missing_returned_at",
        question="What percentage of the 'returned_at' column is missing?",
        datasets={"orders": ORDERS_CSV},
        expected_tools=["get_dataset_profile", "analyze_column", "get_dataset_info"],
        check=lambda r: (
            _answer_contains_number(r["answer"], GT["returned_at_missing_pct"], rel_tol=0.05),
            f"expected ~{GT['returned_at_missing_pct']:.1f}%",
        ),
        ground_truth_note="pandas df['returned_at'].isna().mean() * 100",
    ),
    EvalCase(
        id="top_status",
        question="Which order status has the most orders, and how many?",
        datasets={"orders": ORDERS_CSV},
        expected_tools=["rank_categories", "analyze_column", "compare_categories", "run_sql_query"],
        check=lambda r: (
            GT["top_status"].lower() in r["answer"].lower()
            and _answer_contains_number(r["answer"], GT["top_status_count"], rel_tol=0.02),
            f"expected '{GT['top_status']}' with {GT['top_status_count']}",
        ),
        ground_truth_note="pandas df['status'].value_counts()",
    ),
    EvalCase(
        id="sql_join_top_country",
        question=(
            "Join orders with users_old on user_id, and tell me which "
            "country has the most orders."
        ),
        datasets={"orders": ORDERS_CSV, "users_old": USERS_CSV},
        expected_tools=["run_sql_query"],
        check=lambda r: (
            GT["top_country"].lower() in r["answer"].lower(),
            f"expected '{GT['top_country']}'",
        ),
        ground_truth_note="pandas merge on user_id -> id, value_counts on country",
    ),
    EvalCase(
        id="anomaly_detection_runs",
        question="Are there any anomalies in num_of_item?",
        datasets={"orders": ORDERS_CSV},
        expected_tools=["detect_anomalies"],
        check=None,  # grounding is the point of this case, not a specific number
    ),
    EvalCase(
        id="trend_runs",
        question="Show me the trend of num_of_item over created_at.",
        datasets={"orders": ORDERS_CSV},
        expected_tools=["analyze_trends"],
        check=None,
    ),
    EvalCase(
        id="ml_with_feature_columns",
        question=(
            "Train a model to predict whether an order will be Cancelled "
            "based on num_of_item and gender."
        ),
        datasets={"orders": ORDERS_CSV},
        expected_tools=["train_model"],
        check=lambda r: (
            any(
                entry.get("tool") == "train_model"
                and entry.get("arguments", {}).get("feature_columns")
                for entry in r.get("tool_log", [])
            ),
            "expected train_model called with feature_columns set",
        ),
        ground_truth_note=(
            "regression guard: training on every column instead of just "
            "the two named ones used to hang the API (see README, "
            "Reliability & safety)"
        ),
    ),
    EvalCase(
        id="report_files_exist",
        question="Generate a business report for this dataset.",
        datasets={"orders": ORDERS_CSV},
        expected_tools=["generate_business_report", "deliver_business_report"],
        check=lambda r: (
            bool(r.get("report_files"))
            and all(
                path and Path(path).exists()
                for path in r["report_files"].values()
            ),
            "expected report_files with all 4 files present on disk",
        ),
    ),
    EvalCase(
        id="no_fabrication_on_impossible_column",
        question="What is the average value of the 'profit_margin' column?",
        datasets={"orders": ORDERS_CSV},  # this column does not exist
        expected_tools=[
            "get_dataset_info", "get_dataset_profile", "analyze_column",
            "get_dataset_statistics", "run_sql_query",
        ],
        check=lambda r: (
            "profit_margin" not in r["answer"].lower()
            or any(
                word in r["answer"].lower()
                for word in ("not found", "doesn't exist", "does not exist", "no column", "no such column")
            ),
            "expected the agent to say the column doesn't exist, not invent a value",
        ),
        ground_truth_note="adversarial case: column genuinely absent from orders.csv",
    ),
]


def run_eval(cases: list[EvalCase]) -> list[EvalResult]:
    results = []
    for case in cases:
        print(f"[eval] running: {case.id} ... ", end="", flush=True)
        t0 = time.time()
        try:
            result = run_agent(case.question, case.datasets)
        except Exception as exc:  # the harness itself must not crash on one bad case
            results.append(EvalResult(
                case_id=case.id, question=case.question, tool_grounded=False,
                tools_used=[], factually_correct=None,
                detail=f"agent raised {type(exc).__name__}: {exc}",
                latency_seconds=time.time() - t0, error=str(exc),
            ))
            print("ERROR")
            continue

        elapsed = time.time() - t0
        tools_used = result.get("tools_used", [])
        tool_grounded = any(tool in tools_used for tool in case.expected_tools)

        factually_correct = None
        detail = ""
        if case.check is not None:
            try:
                factually_correct, detail = case.check(result)
            except Exception as exc:
                factually_correct = False
                detail = f"check raised {type(exc).__name__}: {exc}"

        results.append(EvalResult(
            case_id=case.id, question=case.question, tool_grounded=tool_grounded,
            tools_used=tools_used, factually_correct=factually_correct,
            detail=detail, latency_seconds=elapsed, error=result.get("error"),
        ))
        status = "ok" if tool_grounded and factually_correct is not False else "FAIL"
        print(f"{status} ({elapsed:.1f}s)")

    return results


def print_report(results: list[EvalResult]) -> bool:
    print("\n" + "=" * 78)
    print(f"{'Case':<28} {'Tool OK':<8} {'Fact OK':<8} {'Latency':<9} Detail")
    print("-" * 78)

    all_passed = True
    for r in results:
        tool_mark = "PASS" if r.tool_grounded else "FAIL"
        if r.factually_correct is None:
            fact_mark = "n/a"
        else:
            fact_mark = "PASS" if r.factually_correct else "FAIL"

        case_passed = r.tool_grounded and r.factually_correct is not False
        all_passed = all_passed and case_passed

        print(
            f"{r.case_id:<28} {tool_mark:<8} {fact_mark:<8} "
            f"{r.latency_seconds:>6.1f}s  {r.detail}"
        )
        if r.error:
            print(f"{'':<28} error: {r.error}")

    passed_count = sum(
        1 for r in results if r.tool_grounded and r.factually_correct is not False
    )
    print("-" * 78)
    print(f"{passed_count}/{len(results)} cases passed")
    print("=" * 78)
    return all_passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save", default=None,
        help="Optional path to write results as JSON (e.g. eval/eval_results.json)"
    )
    args = parser.parse_args()

    results = run_eval(CASES)
    all_passed = print_report(results)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump([r.__dict__ for r in results], f, indent=2, default=str)
        print(f"\nSaved detailed results to {args.save}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
