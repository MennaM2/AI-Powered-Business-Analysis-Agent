import json
import re
import time

import pandas as pd
from ollama import Client

from app.config import settings

from app.tools.data_tools import (
    get_dataset_info,
    get_dataset_statistics,
    get_dataset_profile
)

from app.tools.analysis_tools import (
    analyze_column,
    compare_categories,
    rank_categories
)

from app.tools.visualization_tools import (
    create_churn_plot,
    create_visualization
)

from app.tools.ml_tools import (
    analyze_churn,
    train_model
)

from app.tools.report_tools import (
    generate_report,
    generate_business_report
)

from app.tools.semantic_search_tools import (
    semantic_search
)

from app.tools.sql_tools import run_sql_query
from app.tools.anomaly_tools import detect_anomalies
from app.tools.trend_tools import analyze_trends, compare_periods
from app.tools.automation_tools import deliver_business_report


MODEL = settings.model

# Keep the model resident in memory between chat() calls in a single
# agent turn (and across turns), so Ollama doesn't reload the weights
# mid-request. On an 8GB machine this matters a lot.
KEEP_ALIVE = "30m"

# Safety cap on the tool-calling loop: if the model somehow never
# settles on a final answer, stop instead of looping forever.
MAX_LLM_CALLS = 8

# A dedicated client with an explicit per-call timeout. The bare
# module-level ollama.chat() has no timeout at all, so a single
# stalled call (network blip, an overloaded cloud endpoint, etc.)
# would otherwise hang the whole request indefinitely instead of
# failing fast with a clear error.
_client = Client(timeout=settings.ollama_timeout_seconds)


def _call_ollama_chat(max_retries: int = 2, **kwargs):
    """Call _client.chat() with a couple of retries on transient
    network failures.

    Production symptom this fixes: a request that had been working
    fine suddenly fails with something like "An existing connection
    was forcibly closed by the remote host" against
    https://ollama.com/api/chat. That is not a bug in this app or a
    sign Ollama is down - it is the remote server (or a proxy in
    between) dropping an idle/long-lived HTTPS connection out from
    under the shared client, which happens occasionally with any
    persistent HTTP client talking to a remote host (this project
    defaults AGENT_MODEL to a "-cloud" suffixed model, which the
    ollama library always sends to https://ollama.com regardless of
    any local host setting). The fix isn't to avoid the call, it's
    to retry it: a fresh connection almost always succeeds
    immediately after a reset like this. If every attempt fails,
    the original exception is re-raised unchanged so the existing
    "AI model call failed" handling still applies.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return _client.chat(**kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                print(
                    f"[Agent] Ollama call failed ({type(exc).__name__}: "
                    f"{exc}) - retrying "
                    f"({attempt + 1}/{max_retries})..."
                )
                time.sleep(1.5 * (attempt + 1))
            continue
    raise last_exc


AVAILABLE_TOOLS = {
    "get_dataset_info": get_dataset_info,
    "get_dataset_statistics": get_dataset_statistics,
    "get_dataset_profile": get_dataset_profile,
    "analyze_column": analyze_column,
    "compare_categories": compare_categories,
    "rank_categories": rank_categories,
    "create_churn_plot": create_churn_plot,
    "create_visualization": create_visualization,
    "analyze_churn": analyze_churn,
    "train_model": train_model,
    "generate_report": generate_report,
    "generate_business_report": generate_business_report,
    "deliver_business_report": deliver_business_report,
    "semantic_search": semantic_search,
    "run_sql_query": run_sql_query,
    "detect_anomalies": detect_anomalies,
    "analyze_trends": analyze_trends,
    "compare_periods": compare_periods
}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_dataset_info",
            "description": (
                "Return dataset shape, dtypes, missing values, "
                "and duplicate row count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string"
                    }
                },
                "required": ["dataset_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_dataset_statistics",
            "description": (
                "Descriptive statistics (mean, std, min, max, etc.) "
                "for numeric columns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string"
                    }
                },
                "required": ["dataset_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_column",
            "description": (
                "Analyze one column: type, missing values, "
                "unique count, and stats."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string"
                    },
                    "column_name": {
                        "type": "string"
                    }
                },
                "required": [
                    "dataset_name",
                    "column_name"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_categories",
            "description": (
                "Compare a numerical or binary target "
                "across categories (count + mean per category)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string"
                    },
                    "category_column": {
                        "type": "string"
                    },
                    "target_column": {
                        "type": "string"
                    }
                },
                "required": [
                    "dataset_name",
                    "category_column",
                    "target_column"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rank_categories",
            "description": (
                "Rank categories (e.g. products or regions) by an "
                "aggregated numeric metric to find top and bottom "
                "performers. Use for 'which products are "
                "underperforming' or 'top 5 regions by revenue'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string"},
                    "category_column": {"type": "string"},
                    "metric_column": {"type": "string"},
                    "aggregation": {
                        "type": "string",
                        "enum": ["sum", "mean", "count", "median"]
                    },
                    "top_n": {"type": "integer"}
                },
                "required": [
                    "dataset_name", "category_column", "metric_column"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_churn_plot",
            "description": (
                "Create a visualization of customer churn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string"
                    }
                },
                "required": ["dataset_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_churn",
            "description": (
                "Analyze a customer churn dataset "
                "using a Random Forest model and return "
                "churn rate and important features."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string"
                    }
                },
                "required": ["dataset_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_dataset_profile",
            "description": (
                "Profile a dataset: column types, missing %, "
                "duplicates, outliers, top correlations, candidate "
                "targets. Use for open-ended requests like "
                "'analyze this dataset'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string"
                    }
                },
                "required": ["dataset_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_visualization",
            "description": (
                "Create and save a chart. Use 'column' for "
                "histogram/bar_chart/box_plot; 'column_x' and "
                "'column_y' for scatter_plot; 'target_column' for "
                "target_distribution; 'category_column' and "
                "'target_column' for category_comparison. "
                "correlation_heatmap needs no columns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string"
                    },
                    "chart_type": {
                        "type": "string",
                        "enum": [
                            "histogram",
                            "bar_chart",
                            "box_plot",
                            "correlation_heatmap",
                            "scatter_plot",
                            "target_distribution",
                            "category_comparison"
                        ]
                    },
                    "column": {
                        "type": "string"
                    },
                    "column_x": {
                        "type": "string"
                    },
                    "column_y": {
                        "type": "string"
                    },
                    "category_column": {
                        "type": "string"
                    },
                    "target_column": {
                        "type": "string"
                    }
                },
                "required": ["dataset_name", "chart_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "train_model",
            "description": (
                "Train a baseline Random Forest to predict "
                "target_column (auto-detects classification vs "
                "regression) and return evaluation metrics plus "
                "top features. Always pass feature_columns when the "
                "user names specific columns to predict from (e.g. "
                "'based on X and Y') - omitting it uses every other "
                "column, including timestamp/id-like text columns, "
                "which can make training extremely slow or time out."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string"
                    },
                    "target_column": {
                        "type": "string"
                    },
                    "task_type": {
                        "type": "string",
                        "enum": ["classification", "regression"]
                    },
                    "feature_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Restrict training to exactly these "
                            "columns as features. Use this whenever "
                            "the user names specific columns."
                        )
                    }
                },
                "required": ["dataset_name", "target_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": (
                "Generate a generic technical report (overview, "
                "quality, stats, correlations, optional ML) as "
                "Markdown+HTML in outputs/. Prefer "
                "generate_business_report for business-framed "
                "requests."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string"
                    },
                    "target_column": {
                        "type": "string"
                    }
                },
                "required": ["dataset_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_business_report",
            "description": (
                "Generate a full business analysis report "
                "(executive summary, key metrics, trends, "
                "anomalies, top/bottom performers, insights, "
                "recommendations) and save it as .md and .html in "
                "outputs/. Auto-detects a date column, a "
                "revenue/sales-like metric column, and a category "
                "column if not given. Use for 'create a business "
                "report' or 'analyze this dataset completely'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string"},
                    "date_column": {"type": "string"},
                    "value_column": {"type": "string"},
                    "category_column": {"type": "string"},
                    "target_column": {"type": "string"}
                },
                "required": ["dataset_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deliver_business_report",
            "description": (
                "The agent's automation action: generates the "
                "business report AND saves it to disk (outputs/), "
                "a real side effect. If 'email_to' is given and "
                "SMTP is configured in the environment, also emails "
                "the HTML report. Use when the user asks to "
                "'generate and save' or 'send' a report."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string"},
                    "date_column": {"type": "string"},
                    "value_column": {"type": "string"},
                    "category_column": {"type": "string"},
                    "target_column": {"type": "string"},
                    "email_to": {"type": "string"}
                },
                "required": ["dataset_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Search a free-text column by meaning (vector "
                "similarity), not exact keywords. Use for requests "
                "like 'find rows about X' or 'which entries mention "
                "Y' on a text column such as reviews, comments, or "
                "descriptions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string"
                    },
                    "text_column": {
                        "type": "string"
                    },
                    "query": {
                        "type": "string"
                    },
                    "top_k": {
                        "type": "integer"
                    }
                },
                "required": ["dataset_name", "text_column", "query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": (
                "Run a read-only SQL SELECT query. Every uploaded "
                "dataset in this session is available as its own "
                "table (see the table names and columns listed in "
                "the dataset catalog above) - JOIN across them by "
                "table name for questions spanning multiple "
                "datasets. Only SELECT/WITH queries are allowed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": (
                            "e.g. SELECT o.status, COUNT(*) FROM "
                            "orders o JOIN users_old u ON o.user_id "
                            "= u.id GROUP BY o.status"
                        )
                    }
                },
                "required": ["sql_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": (
                "Detect unusual/outlier values. Pass 'column' to "
                "check a single numeric column (e.g. revenue) using "
                "IQR + Z-score; omit it to run multivariate anomaly "
                "detection (Isolation Forest) across all numeric "
                "columns at once. Optionally pass 'date_column' to "
                "include dates on flagged rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string"},
                    "column": {"type": "string"},
                    "date_column": {"type": "string"}
                },
                "required": ["dataset_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_trends",
            "description": (
                "Analyze how a numeric column changes over time. "
                "Resamples by the given frequency and returns the "
                "period-by-period series, overall direction, total "
                "% change, and the largest single-period rise/drop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string"},
                    "date_column": {"type": "string"},
                    "value_column": {"type": "string"},
                    "freq": {
                        "type": "string",
                        "enum": ["day", "week", "month", "quarter", "year"]
                    }
                },
                "required": ["dataset_name", "date_column", "value_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_periods",
            "description": (
                "Compare a numeric column between two time periods "
                "(e.g. this month vs last month). Omit period_a/"
                "period_b to auto-compare the two most recent "
                "periods. Pass 'group_column' (e.g. region or "
                "product) to also rank which groups grew or "
                "declined the most - use this for 'which region "
                "grew the most?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string"},
                    "date_column": {"type": "string"},
                    "value_column": {"type": "string"},
                    "freq": {
                        "type": "string",
                        "enum": ["day", "week", "month", "quarter", "year"]
                    },
                    "period_a": {
                        "type": "string",
                        "description": "e.g. '2024-05' for a month"
                    },
                    "period_b": {"type": "string"},
                    "group_column": {"type": "string"}
                },
                "required": ["dataset_name", "date_column", "value_column"]
            }
        }
    }
]


SYSTEM_PROMPT = """
You are an AI-Powered Business Analysis & Automation Agent.

You answer business questions about the user's uploaded dataset(s) by
calling tools and reasoning over their real output.

The user's message is preceded by a dataset catalog listing every
table currently available in this session (name, row count, columns).
There may be one dataset or several related ones (e.g. orders,
order_items, users). For any tool other than run_sql_query, pass the
exact table name from that catalog as the 'dataset_name' argument -
if only one dataset is uploaded you may omit it. run_sql_query
automatically has every table available and can JOIN across them by
name; use it whenever a question spans more than one dataset (e.g.
"revenue by user country" needs orders/order_items joined to users).

Rules:
1. Never invent dataset values, metrics, or findings. Every number
   you state must come from a tool result.
2. Always use tools for factual dataset information; treat tool
   results as the authoritative source and never contradict them.
3. Break open-ended business questions ("why did revenue drop?")
   into a short sequence of tool calls: inspect/profile the data,
   analyze the relevant trend or comparison, check top/bottom
   performers or anomalies if relevant, then explain. Do not
   hard-code one fixed workflow - choose tools based on the
   question and what earlier tool results show.
4. Call the minimum number of tools needed to answer confidently.
   Do not call additional tools once you already have enough.
5. If a question needs a column that does not clearly exist in the
   dataset, say so explicitly instead of guessing a column name.
6. Never construct, modify, or guess filesystem paths - the
   application provides the trusted dataset path.
7. Distinguish correlation and feature importance from causation.
8. If earlier turns in this conversation are shown, use them to
   resolve follow-up references (e.g. "that product", "the worst
   region") - but still call a tool to get the current facts rather
   than reusing a stale number from memory.
9. Never claim an automation action (saving or emailing a report)
   succeeded unless the tool result actually confirms it did.
10. Every tool listed for you is real and working right now - never
    claim a tool is unavailable, broken, not implemented, or not
    supported. If you are unsure whether to use one, call it and let
    the real result guide you, rather than guessing or refusing.
11. Never state a fact, number, or finding about the data - in any
    form, including a table - unless it came from a tool call you
    actually made this turn. If you have not called a tool yet and
    the question needs data, call one now instead of answering; it
    is always fine to ask a short clarifying question back to the
    user instead if the request is genuinely ambiguous.
12. Explain findings clearly and in depth, in business language, not
    just raw numbers - note what a finding means in practice.
13. Structure longer answers with short bold headers or bullet
    points so they are easy to scan; weave multiple related facts
    into one coherent narrative rather than a flat list of numbers.
14. If information is unavailable or a tool returns an error, say so
    explicitly rather than filling the gap yourself.
15. Do not reveal your internal step-by-step reasoning - call the
    tools you need, then give a clear final answer.
15. Only call a tool through the real function-calling mechanism.
    Never write out what a tool call or its result would look like
    as plain text - if you need a tool's output, call it for real
    and wait for the actual result before continuing.
"""


# ---------------------------------------------------------------------
# Harmony-format leak guard
# ---------------------------------------------------------------------
# gpt-oss (and other Harmony-format models) internally separate a
# hidden "analysis" channel from the "final" answer channel using
# control tokens like <|channel|>final<|message|>...<|end|>. The
# client is supposed to strip these and hand back only the final
# channel's text. Over some Ollama Cloud responses this stripping
# doesn't happen and the raw tokens leak into response.message.content.
#
# This matters beyond cosmetics: an unsanitized answer gets stored in
# conversation memory and re-sent as context on the next turn, which
# can confuse - or even 500 - a subsequent call. So every answer is
# cleaned before it's returned or remembered.
_CHANNEL_FINAL_PATTERN = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)"
    r"(?:<\|end\|>|<\|return\|>|<\|call\|>|$)",
    re.DOTALL,
)
_ANY_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^|]*\|>")


def _clean_model_text(text: str) -> str:
    """
    If the raw Harmony channel tokens leaked into `text`, keep only
    the last 'final' channel's content. Otherwise, strip any stray
    <|...|> control tokens as a safety net. Plain, well-formed text
    is returned unchanged.
    """
    if not text or "<|" not in text:
        return text

    final_channel_matches = _CHANNEL_FINAL_PATTERN.findall(text)
    if final_channel_matches:
        return final_channel_matches[-1].strip()

    return _ANY_SPECIAL_TOKEN_PATTERN.sub("", text).strip()


# ---------------------------------------------------------------------
# Fabricated tool-call narration guard
# ---------------------------------------------------------------------
# A more serious failure mode than the bracket-token leak above: the
# model can narrate what LOOKS like a real function call and its
# result - e.g. "commentary to=functions.rank_categories json{...}
# functions.rank_categoriescommentary{...fake numbers...}" - entirely
# as free text, without ever populating response.message.tool_calls.
# Our loop only executes *real* tool_calls, so nothing here is real:
# the "result" is the model inventing plausible-looking numbers. This
# has been observed in production returning category values that
# don't even exist in the actual dataset. Never show this to the
# user as if it were a verified answer.
_FAKE_TOOLCALL_PATTERN = re.compile(
    r"to=functions\.\w+|functions\.\w+(?:commentary|json)",
    re.IGNORECASE,
)


def _looks_like_fabricated_tool_call(text: str) -> bool:
    return bool(text) and bool(_FAKE_TOOLCALL_PATTERN.search(text))


# ---------------------------------------------------------------------
# False tool-refusal guard
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Zero-tool-call guard (unified)
# ---------------------------------------------------------------------
# Chasing every possible refusal/fabrication phrasing with regex was
# a losing game - proven repeatedly in production: "not functioning"
# -> "not implemented" -> "SQL engine isn't available" -> a clean
# fabricated table with no refusal language at all. Every one of
# those slipped past a text pattern built for the previous one.
#
# The actual invariant that matters isn't WHAT the model's excuse
# says - it's whether a real tool was called. If tool_count is 0 for
# the whole turn, the model has no real data behind whatever it just
# said, full stop, regardless of phrasing. So instead of pattern-
# matching the text, we structurally require at least one real tool
# call before accepting a final answer - unless the response is
# clearly just a short clarifying question or greeting that plainly
# doesn't claim any data.
MAX_REFUSAL_RETRIES = 2

# A "safe" zero-tool answer: a short clarifying question back to the
# user, or a short greeting/meta reply - neither claims data, so both
# are fine without ever touching a tool.
_SAFE_SHORT_ANSWER_MAX_CHARS = 300


def _is_safe_zero_tool_response(text: str) -> bool:
    """
    Deliberately narrow, on purpose: after the "short + no digits"
    exception let an actual refusal ("The tool is currently not
    functioning.") through as "safe" (no digits, no report-language
    keywords - but still a refusal), the only rule kept is the one
    that can't be gamed by rephrasing: is this literally a question
    back to the user? A real clarifying question always ends in "?".
    Anything else with zero tool calls - a refusal, a fabricated
    table, a flat statement of "facts" - gets rejected, regardless of
    how short, plain, or reasonable-sounding it is. A plain greeting
    with no dataset question will occasionally get an unnecessary
    retry; that's a acceptable cost against ever again taking a false
    "sorry, that's unavailable" at face value.
    """
    if not text:
        return False
    stripped = text.strip()
    return len(stripped) < _SAFE_SHORT_ANSWER_MAX_CHARS and stripped.endswith("?")


def _json_safe(value):
    """
    Recursively replace NaN/Infinity/-Infinity with None.

    Python's json.dumps allows these by default and emits the
    literal tokens NaN/Infinity, which is not valid JSON per spec.
    A tool result containing one (e.g. an R2 or accuracy metric that
    came out NaN on an edge-case dataset) would otherwise get sent
    to the LLM API as malformed JSON inside the tool message - a
    plausible cause of a cloud-side 500 on the following call.
    """
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _build_dataset_catalog(datasets: dict) -> str:
    """
    A short, per-turn description of every dataset available in this
    session - table name, row count, and columns - so the model
    knows what 'dataset_name' values are valid for single-dataset
    tools, and what table names it can JOIN in run_sql_query. Built
    fresh each turn (cheap: header-only read) rather than cached, so
    it's always accurate even right after a new upload.
    """
    if not datasets:
        return "No datasets are currently uploaded in this session."

    lines = ["Available datasets in this session:"]
    for table_name, file_path in datasets.items():
        try:
            preview = pd.read_csv(file_path, nrows=0)
            columns = ", ".join(preview.columns.tolist())
            row_count = sum(1 for _ in open(file_path, encoding="utf-8")) - 1
            lines.append(
                f"- '{table_name}' (~{max(row_count, 0)} rows): {columns}"
            )
        except Exception:
            lines.append(f"- '{table_name}': (could not be previewed)")

    return "\n".join(lines)


def _format_real_result_fallback(last_successful_result: dict) -> str:
    """
    Render an already-executed, genuinely real tool result as plain
    text - used only as a fallback when the model's own final answer
    got discarded for looking fabricated, so the user still gets the
    real data instead of nothing.
    """
    if not last_successful_result:
        return ""

    tool_name = last_successful_result["tool"]
    result = last_successful_result["result"]

    if tool_name == "run_sql_query" and "rows" in result:
        lines = [
            f"Real result from run_sql_query "
            f"(query: {result.get('query')}):",
            "",
        ]
        rows = result.get("rows", [])[:10]
        if rows:
            columns = result.get("columns") or list(rows[0].keys())
            lines.append(" | ".join(str(c) for c in columns))
            for row in rows:
                lines.append(" | ".join(str(row.get(c, "")) for c in columns))
        if result.get("truncated"):
            lines.append(
                f"... ({result.get('row_count')} rows total, "
                f"showing first {len(rows)})"
            )
        return "\n".join(lines)

    try:
        dumped = json.dumps(_json_safe(result), indent=2, default=str)
    except Exception:
        dumped = str(result)

    if len(dumped) > 1500:
        dumped = dumped[:1500] + "\n... (truncated)"

    return f"Real result from {tool_name}:\n{dumped}"


def run_agent(
    user_message: str,
    datasets: dict,
    history: list = None
):
    """
    Run one agent turn.

    `datasets` maps table_name -> CSV file path for every dataset
    currently uploaded in this session (one or many). Single-dataset
    tools are called with a `dataset_name` argument the model picks
    from this catalog; run_sql_query gets every table at once and
    can JOIN across them.

    `history`, if given, is a list of {"role": "user"|"assistant",
    "content": str} dicts from earlier turns in this session, used so
    the agent can resolve follow-up questions.
    """

    datasets = {
        name: path.replace("\\", "/") for name, path in (datasets or {}).items()
    }

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    for turn in (history or []):
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({
                "role": turn["role"],
                "content": turn["content"]
            })

    messages.append({
        "role": "user",
        "content": (
            f"{_build_dataset_catalog(datasets)}\n\n"
            f"User request: {user_message}"
        )
    })

    total_start = time.perf_counter()
    tool_count = 0
    tools_used = []
    tool_log = []
    last_plot_path = None
    last_report_files = None

    # Tracks the most recent *real*, successfully executed tool
    # result (not model text) - so if the model's final answer gets
    # discarded for looking fabricated, we can still hand back the
    # genuine data we already have instead of nothing at all.
    last_successful_result = None
    last_failed_result = None
    successful_tools = []

    # How many times we've caught the model falsely refusing a tool
    # call and forced it to retry (see _looks_like_false_tool_refusal).
    refusal_retry_count = 0

    # Per-stage timings, in call order, so we can see exactly where
    # time is going instead of guessing.
    stage_timings = []
    llm_call_number = 0

    while True:

        llm_call_number += 1

        if llm_call_number > MAX_LLM_CALLS:
            total_time = time.perf_counter() - total_start
            print(
                f"[Agent] stopped after reaching MAX_LLM_CALLS="
                f"{MAX_LLM_CALLS}"
            )
            return {
                "answer": (
                    "I wasn't able to reach a final answer within the "
                    "allowed number of tool calls for this question. "
                    "Please try asking something more specific."
                ),
                "plot_path": last_plot_path,
                "report_files": last_report_files,
                "tools_used": tools_used,
                "tool_log": tool_log,
                "error": "max_iterations_reached",
                "latency": {
                    "total_seconds": round(total_time, 2),
                    "tool_calls": tool_count,
                    "stages": stage_timings
                }
            }

        model_start = time.perf_counter()

        # Previous assumption ("tool selection is always reliable, so
        # only force thinking after a tool call") was based on
        # limited evidence and just got contradicted: an empty
        # response happened on a first call, no tool attempted at
        # all. Rather than keep tuning this blind, thinking is now
        # always on - the safer, less clever default - and every
        # anomaly branch below logs the model's raw, uncleaned output
        # (including the separate .thinking field) so any further
        # failure leaves real diagnostic evidence server-side instead
        # of another guess.
        allow_thinking = True

        try:
            response = _call_ollama_chat(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                think=allow_thinking,
                keep_alive=KEEP_ALIVE,
                options={
                    "num_ctx": 8192,
                    "temperature": 0,
                    "num_predict": 1024
                }
            )
        except Exception as exc:
            total_time = time.perf_counter() - total_start
            print(f"[Agent] LLM call failed: {exc}")
            return {
                "answer": (
                    "The AI model call failed, so I couldn't process "
                    "this request. Please confirm Ollama is running "
                    "and reachable, then try again.\n\n"
                    f"Details: {type(exc).__name__}: {exc}"
                ),
                "plot_path": last_plot_path,
                "report_files": last_report_files,
                "tools_used": tools_used,
                "tool_log": tool_log,
                "error": "llm_unavailable",
                "error_detail": f"{type(exc).__name__}: {exc}",
                "latency": {
                    "total_seconds": round(total_time, 2),
                    "tool_calls": tool_count,
                    "stages": stage_timings
                }
            }

        model_time = time.perf_counter() - model_start

        stage_timings.append({
            "stage": f"llm_call_{llm_call_number}",
            "seconds": round(model_time, 2)
        })

        print(
            f"[LLM] call completed in {model_time:.2f}s"
        )

        messages.append(response.message)

        if not response.message.tool_calls:

            total_time = (
                time.perf_counter() - total_start
            )

            print(
                f"[Agent] total latency: {total_time:.2f}s "
                f"| tool_calls={tool_count}"
            )

            cleaned_answer = _clean_model_text(response.message.content)
            is_empty = not cleaned_answer or not cleaned_answer.strip()
            is_fabricated = (
                not is_empty
                and _looks_like_fabricated_tool_call(cleaned_answer)
            )
            needs_retry = (
                not is_empty
                and not is_fabricated
                and not successful_tools
                and refusal_retry_count < MAX_REFUSAL_RETRIES
                and not _is_safe_zero_tool_response(cleaned_answer)
            )

            if is_empty or is_fabricated or needs_retry:
                # Real diagnostic evidence, not a guess: the exact raw
                # fields the model returned this call, before any
                # cleaning. If this keeps happening, these server
                # logs (not the user-facing message) are what should
                # drive the next fix.
                print(
                    "[Agent][DIAGNOSTIC] anomaly on llm_call_"
                    f"{llm_call_number} (tool_calls_so_far={tool_count}, "
                    f"allow_thinking={allow_thinking}):\n"
                    f"  raw content = {response.message.content!r}\n"
                    f"  thinking    = "
                    f"{getattr(response.message, 'thinking', None)!r}\n"
                    f"  tool_calls  = {response.message.tool_calls!r}"
                )

            if needs_retry:
                refusal_retry_count += 1
                print(
                    "[Agent] No successful tool call yet this turn and "
                    "the answer doesn't look like a safe clarifying "
                    "question - forcing a retry regardless of how the "
                    f"answer is worded ({refusal_retry_count}/"
                    f"{MAX_REFUSAL_RETRIES})."
                )
                if last_failed_result:
                    # A tool WAS called but it errored (e.g. a bad
                    # JOIN column or table name) and the model gave up
                    # with an unverified excuse instead of fixing the
                    # query. Feed the real error back so the retry has
                    # something concrete to correct, instead of
                    # repeating a vague "call a tool" nudge that
                    # doesn't address what actually went wrong.
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your call to '{last_failed_result['tool']}' "
                            f"failed with this real error: "
                            f"{last_failed_result['error']} "
                            "Do not tell the user the tool is "
                            "unavailable or not implemented - it is "
                            "real and working, your arguments (e.g. "
                            "table/column names for a JOIN) were "
                            "wrong. Check the dataset catalog above "
                            "for the exact table and column names, "
                            "fix the query, and call the tool again "
                            "via function calling. Do not answer with "
                            "claims, numbers, or explanations that are "
                            "not backed by a tool call that actually "
                            "succeeded."
                        )
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": (
                            "You have not called any tool yet this turn. "
                            "Every tool listed for you is real and "
                            "working - call the one that fits this "
                            "request via function calling and answer "
                            "using its actual result. Do not answer with "
                            "claims, numbers, or explanations that are "
                            "not backed by a tool call you actually made."
                        )
                    })
                continue

            if is_empty or is_fabricated or (
                not successful_tools and not _is_safe_zero_tool_response(cleaned_answer)
            ):
                if is_empty:
                    print(
                        "[Agent] Model returned an empty final answer "
                        "(no content, no tool call)."
                    )
                    reason = (
                        "The model returned an empty response for "
                        "this request."
                    )
                    error_code = "empty_model_response"
                elif is_fabricated:
                    print(
                        "[Agent] Discarded a final answer that narrated "
                        "an un-executed tool call instead of using real "
                        "function calling - refusing to show possibly "
                        "fabricated numbers."
                    )
                    reason = (
                        "I wasn't able to produce a verified final "
                        "answer for this request - part of the "
                        "model's response looked like a narrated, "
                        "un-executed tool call rather than a real "
                        "result, so I'm not showing it to avoid "
                        "presenting numbers that were never actually "
                        "computed."
                    )
                    error_code = "unreliable_model_output"
                else:
                    print(
                        "[Agent] Discarded a final answer with zero "
                        "tool calls this turn, after exhausting the "
                        "retry budget - cannot verify it against real "
                        "data no matter how it's phrased."
                    )
                    reason = (
                        "I wasn't able to verify this answer - no "
                        "tool was actually called this turn, so "
                        "there's no real data behind it, regardless "
                        "of how the response reads. I'm not showing "
                        "it to avoid presenting unverified numbers."
                    )
                    error_code = "unverified_data_claim"


                if successful_tools:
                    tools_note = (
                        "These tools *did* run successfully and their "
                        f"results are real: {', '.join(successful_tools)}."
                    )
                elif last_failed_result:
                    tools_note = (
                        f"'{last_failed_result['tool']}' was called "
                        f"but returned an error: "
                        f"{last_failed_result['error']}"
                    )
                elif tools_used:
                    tools_note = (
                        f"'{', '.join(tools_used)}' was attempted but "
                        "did not return a usable result."
                    )
                else:
                    tools_note = (
                        "No tools were called this turn."
                    )

                real_data_block = _format_real_result_fallback(
                    last_successful_result
                )

                fallback_answer = f"{reason} {tools_note}"

                if real_data_block:
                    fallback_answer += (
                        "\n\nHere is the actual, verified data from "
                        f"that tool call:\n\n{real_data_block}"
                    )
                elif last_failed_result:
                    fallback_answer += (
                        " Try adjusting the request based on that "
                        "error (e.g. a different column name)."
                    )
                else:
                    fallback_answer += (
                        " Please try rephrasing, or ask about one "
                        "specific metric at a time."
                    )

                return {
                    "answer": fallback_answer,
                    "plot_path": last_plot_path,
                    "report_files": last_report_files,
                    "tools_used": tools_used,
                    "tool_log": tool_log,
                    "error": error_code,
                    "latency": {
                        "last_model_seconds": round(model_time, 2),
                        "total_seconds": round(total_time, 2),
                        "tool_calls": tool_count,
                        "stages": stage_timings
                    }
                }

            return {
                "answer": cleaned_answer,
                "plot_path": last_plot_path,
                "report_files": last_report_files,
                "tools_used": tools_used,
                "tool_log": tool_log,
                "latency": {
                    "last_model_seconds": round(
                        model_time,
                        2
                    ),
                    "total_seconds": round(
                        total_time,
                        2
                    ),
                    "tool_calls": tool_count,
                    "stages": stage_timings
                }
            }

        for tool_call in response.message.tool_calls:

            tool_name = tool_call.function.name
            arguments = dict(tool_call.function.arguments)

            tool_count += 1
            if tool_name not in tools_used:
                tools_used.append(tool_name)

            tool_start = time.perf_counter()

            if tool_name not in AVAILABLE_TOOLS:
                result = {"error": f"Unknown tool '{tool_name}'."}
            elif tool_name == "run_sql_query":
                try:
                    result = run_sql_query(
                        datasets=datasets,
                        sql_query=arguments.get("sql_query", "")
                    )
                except Exception as exc:
                    result = {"error": str(exc)}
            else:
                dataset_name = arguments.pop("dataset_name", None)

                if dataset_name and dataset_name in datasets:
                    resolved_path = datasets[dataset_name]
                elif dataset_name:
                    resolved_path = None
                elif len(datasets) == 1:
                    # Only one dataset uploaded - unambiguous, no
                    # need to force the model to name it every time.
                    resolved_path = next(iter(datasets.values()))
                else:
                    resolved_path = None

                if resolved_path is None:
                    result = {
                        "error": (
                            f"Unknown or missing dataset_name "
                            f"'{dataset_name}'. Available datasets: "
                            f"{list(datasets.keys())}. Pass one of "
                            f"these exactly as 'dataset_name'."
                        )
                    }
                else:
                    try:
                        tool_function = AVAILABLE_TOOLS[tool_name]
                        arguments["file_path"] = resolved_path
                        result = tool_function(**arguments)
                    except TypeError as exc:
                        result = {
                            "error": (
                                f"Invalid arguments for '{tool_name}': {exc}"
                            )
                        }
                    except Exception as exc:
                        result = {
                            "error": str(exc)
                        }

            if (
                tool_name in ("create_visualization", "create_churn_plot")
                and isinstance(result, dict)
                and "error" not in result
                and "file_path" in result
            ):
                last_plot_path = result["file_path"]

            if (
                tool_name in (
                    "generate_report",
                    "generate_business_report",
                    "deliver_business_report",
                )
                and isinstance(result, dict)
                and "error" not in result
            ):
                # Surface the saved report files the same way
                # last_plot_path surfaces a chart - so the chat UI
                # can offer download buttons for a report generated
                # from a plain-language chat question, not just from
                # the dedicated Report Generation form.
                last_report_files = {
                    "markdown": result.get("report_path_markdown"),
                    "html": result.get("report_path_html"),
                    "pdf": result.get("report_path_pdf"),
                    "docx": result.get("report_path_docx"),
                }

            if isinstance(result, dict) and "error" not in result:
                last_successful_result = {
                    "tool": tool_name,
                    "result": result
                }
                if tool_name not in successful_tools:
                    successful_tools.append(tool_name)
            elif isinstance(result, dict) and "error" in result:
                last_failed_result = {
                    "tool": tool_name,
                    "error": result["error"]
                }

            tool_time = (
                time.perf_counter()
                - tool_start
            )

            print(
                f"[Tool] {tool_name} "
                f"completed in {tool_time:.2f}s"
            )

            stage_timings.append({
                "stage": f"tool:{tool_name}",
                "seconds": round(tool_time, 2)
            })

            tool_log.append({
                "tool": tool_name,
                "arguments": {
                    key: value for key, value in arguments.items()
                    if key != "file_path"
                },
                "success": (
                    isinstance(result, dict) and "error" not in result
                )
            })

            messages.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": json.dumps(
                    _json_safe(result),
                    default=str,
                    allow_nan=False
                )
            })