# AI-Powered Business Analysis & Automation Agent

An agentic backend + UI that turns natural-language **business questions**
about an uploaded CSV into real tool calls — SQL, statistics, trend
analysis, anomaly detection, visualization, machine learning, and
automated report generation/delivery — and reasons over the actual
results, not invented ones.

> This project is an upgrade of an earlier **AI Data Analyst Agent**
> (dataset profiling + chat). That system is preserved and extended
> here, not replaced: every original tool still works, and 7 new
> tools + a FastAPI backend + tests were added on top of it.

---

## Demo

Two short walkthroughs:

| | |
|---|---|
| **1. Uploading a dataset** | `docs/demo-upload.mp4` |
| **2. Asking business questions** | `docs/demo-questions.mp4` |

> GitHub doesn't play a video from a plain relative link like the
> table above — it only auto-embeds a video that was uploaded
> *through* GitHub's own editor. To make these actually playable on
> the repo page: open this file in the GitHub web editor (or a new
> Issue/PR comment), drag each `.mp4` into the text box, and GitHub
> will replace it with a `https://github.com/user-attachments/...`
> link that plays inline. Paste those two links in place of the
> table above once you've done that, then delete this note.

---

## 1. What this is, and why it isn't "just a chatbot"

A chatbot answers from what the model already knows. This agent
doesn't know anything about your data — it has to go find out, every
time:

```
"Why did revenue drop last month?"
        ↓
Agent inspects the dataset, calls analyze_trends and compare_periods,
sees March is down 34%, calls detect_anomalies to check for a data
error, calls rank_categories to see which product/region drove it,
then explains — citing only numbers those tool calls actually returned.
```

Every number in a response is traceable to a tool call in that turn's
`tool_log`. If a tool errors or a column doesn't exist, the agent says
so — it does not fill the gap with a plausible-sounding guess. The
one automation tool (`deliver_business_report`) only ever reports
success after confirming the output files exist on disk (and, for
email, after `smtplib` confirms the send didn't raise).

---

## 2. Architecture

```
Streamlit UI (app.py)                 -- thin HTTP client, no agent logic
        |  HTTP
        v
FastAPI backend (app/api/main.py)     -- /health /upload /chat /analyze /report
        |
        v
Agent Orchestrator (app/agent/agent.py)
        |
        v
Ollama (gpt-oss:20b-cloud, real tool-calling, think=False)
        |
        v
Tool Layer (app/tools/*.py)
    |-- data_tools.py          -> get_dataset_info, get_dataset_statistics, get_dataset_profile
    |-- analysis_tools.py      -> analyze_column, compare_categories, rank_categories
    |-- sql_tools.py           -> run_sql_query            (NEW - DuckDB, read-only)
    |-- anomaly_tools.py       -> detect_anomalies          (NEW - IQR+Z-score / Isolation Forest)
    |-- trend_tools.py         -> analyze_trends, compare_periods  (NEW)
    |-- visualization_tools.py -> create_visualization, create_churn_plot
    |-- ml_tools.py            -> train_model, analyze_churn
    |-- semantic_search_tools.py -> semantic_search (FAISS + sentence-transformers)
    |-- report_tools.py        -> generate_report, generate_business_report (NEW)
    `-- automation_tools.py    -> deliver_business_report    (NEW - real save + optional email)
        |
        v
Final business-language answer, with tools_used / tool_log / latency
returned to the API and shown in the UI.
```

Separation of concerns: agent logic (`app/agent/`), tools
(`app/tools/`), API layer (`app/api/`), UI (`app.py`), config
(`app/config.py`) — no business logic lives inside `app.py` anymore.

### Agent workflow

The system prompt does **not** hard-code a fixed pipeline per question
type. It instructs the model to break a question into the minimum
sequence of tool calls needed, e.g.:

```
"Why did revenue drop last month?"
   -> get_dataset_profile (understand the data)
   -> analyze_trends / compare_periods (confirm the drop, find when)
   -> detect_anomalies (rule out a data error)
   -> rank_categories (find which product/region drove it)
   -> final answer, in business language
```

### Conversation memory

`app/agent/memory.py` keeps a lightweight, in-process session (dataset
path + a trimmed rolling window of turns) so follow-ups work:

```
"Analyze sales.csv."             -> agent profiles it
"Which product performed worst?" -> agent resolves "product" from context, ranks
"Create a chart for it."         -> agent resolves "it" to that product
```

Sessions - datasets and full chat history - are persisted to SQLite
(`app/agent/memory.py`), not held in process memory. That means:
uploading another file adds a table alongside the ones already there
(nothing gets wiped), a chat survives an API restart, and past chats
can be listed and resumed via `GET /sessions` / `GET /sessions/{id}`.

Each session gets an auto-generated `title`: the dataset name at
upload time as a placeholder, replaced by the first question once
one's been asked (`app/api/main.py`, `_title_from_text`) — the same
way ChatGPT/Claude name a conversation from its first message. The
Streamlit sidebar's "Resume a chat" renders these as a clickable
list (one button per session, active one highlighted) instead of a
dropdown + separate "Load" button, so it reads like a normal chat
history panel rather than a picker.

### Multiple datasets and SQL joins

A session can hold more than one uploaded CSV at once, each exposed
as its own table (name derived from the filename, e.g.
`orders.csv` → `orders`). `run_sql_query` registers every table in
the session simultaneously, so the model can JOIN across them:

```sql
SELECT u.country, SUM(oi.sale_price) AS revenue
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
JOIN users u  ON o.user_id  = u.id
GROUP BY u.country
ORDER BY revenue DESC
LIMIT 5
```

Every other (single-dataset) tool takes a `dataset_name` argument the
model picks from a per-turn dataset catalog (table name, row count,
columns) injected into the conversation - if only one dataset is
uploaded, `dataset_name` can be omitted and it resolves automatically.

**Limitation:** SQLite is a single-file, single-process store - fine
for a portfolio deployment, but a multi-worker production setup would
move this to Postgres/Redis instead.

---

## 3. Available tools

| Tool | Purpose |
|---|---|
| `get_dataset_info` | shape, dtypes, missing values, duplicates |
| `get_dataset_statistics` | descriptive stats for numeric columns |
| `get_dataset_profile` | full profile: types, missing %, outliers, correlations, candidate targets |
| `analyze_column` | single-column deep dive |
| `compare_categories` | count + mean of a target per category |
| `rank_categories` **(new)** | top/bottom performers by an aggregated metric |
| `run_sql_query` **(new)** | read-only SQL (DuckDB) against the dataset as table `dataset` |
| `detect_anomalies` **(new)** | IQR+Z-score (one column) or Isolation Forest (multivariate) |
| `analyze_trends` **(new)** | period-by-period trend, direction, biggest swings |
| `compare_periods` **(new)** | period A vs B, with optional per-group growth/decline ranking |
| `create_visualization` | histogram, bar, box, scatter, heatmap, target distribution, category comparison |
| `create_churn_plot` | churn-specific visualization |
| `train_model` | baseline Random Forest (classification or regression) + feature importance |
| `analyze_churn` | churn-specific model + churn rate |
| `semantic_search` | vector search over a free-text column (FAISS + sentence-transformers) |
| `generate_report` | generic technical report (kept for backward compatibility) |
| `generate_business_report` **(new)** | Executive Summary -> Recommendations, auto-detects date/metric/category columns, saved as Markdown, HTML, **PDF, and Word (.docx)** |
| `deliver_business_report` **(new)** | the automation action: saves the report to disk in all four formats, optionally emails it with the PDF attached |

### Tool calling

This is real function calling via Ollama's `tools=` parameter, not a
single LLM turn that free-writes fake JSON. The loop in
`app/agent/agent.py`:

```
while True:
    call the model with the running message history + tool schemas
    if the model didn't request a tool  -> return its answer
    for each requested tool call:
        execute the real Python function
        append the real result to the message history
    (loop back, capped at MAX_LLM_CALLS=8 to avoid runaway loops)
```

### SQL safety

`run_sql_query` validates before anything reaches DuckDB:
- must start with `SELECT` or `WITH ... SELECT`
- a single statement only (semicolon-chained statements rejected)
- a keyword blocklist catches `INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/
  ATTACH/COPY/PRAGMA/...` anywhere in the query, including inside a
  CTE or subquery

This blocklist is intentionally conservative: a `SELECT` that filters
on a string literal containing a blocked word (e.g. `WHERE
status = 'DROPPED'`) will be rejected too. That's a deliberate
fail-safe tradeoff, not a bug.

### Business report

`generate_business_report` auto-detects a date column, a revenue/
sales-like metric column, and a category (region/product/segment)
column when not given, then composes:

**Executive Summary -> Key Metrics -> Trends -> Anomalies ->
Top/Bottom Performers -> Insights -> Recommendations -> Limitations**

as `.md`, `.html`, `.pdf`, and `.docx` in `outputs/`. Every bullet is
built with plain Python string templates around real tool output —
nothing here is LLM-generated free text, so nothing can be
hallucinated. The PDF and DOCX are rendered from that same markdown
by `_markdown_to_pdf`/`_markdown_to_docx` (fpdf2 / python-docx —
pure Python, no system-level dependencies like Cairo/Pango), with
colored section headings, a divider rule, and an indented bullet
block so the document reads like a report rather than a text dump.
Whichever of the four formats exist are also surfaced as download
buttons in the Streamlit UI — both from the dedicated "Report
Generation" form and from a plain chat question that triggers the
same tool.

### Automation

`deliver_business_report` is the one real "does something" action:
1. Generates the report (all four formats).
2. Confirms the `.md`/`.html`/`.pdf`/`.docx` files actually exist on
   disk before reporting `"action": "report_saved"`.
3. If `email_to` is given **and** SMTP is configured via environment
   variables, sends the HTML report via `smtplib` with the **PDF
   attached**. If SMTP isn't configured, or the send fails, the
   response says so explicitly — it never claims an email was sent
   that wasn't.

---

## 4. Structured outputs

Rather than asking the LLM to also emit a JSON envelope (which risks
the model inventing fields under pressure), structural metadata is
derived from real orchestration state, not model text:

```json
{
  "answer": "free-text final answer from the LLM",
  "tools_used": ["analyze_trends", "detect_anomalies"],
  "tool_log": [
    {"tool": "analyze_trends", "arguments": {"...": "..."}, "success": true}
  ],
  "plot_path": "outputs/chart_....png",
  "report_files": {
    "markdown": "outputs/business_report.md",
    "html": "outputs/business_report.html",
    "pdf": "outputs/business_report.pdf",
    "docx": "outputs/business_report.docx"
  },
  "latency_seconds": 4.21,
  "error": null
}
```

`tools_used` / `tool_log` come from the agent's own bookkeeping of
which functions it actually called and whether they errored — never
from asking the model to self-report. Internal chain-of-thought is
never exposed (`think=False` on every Ollama call, and the API only
returns the final answer + tool telemetry).

---

## 5. FastAPI backend

```
GET    /health           -> { status, ollama_reachable, model }
POST   /upload            -> multipart CSV upload (?session_id= optional) -> adds a
                              dataset to that session, or creates a new session if omitted
POST   /chat               -> { session_id, message } -> agent turn, persisted history
POST   /analyze             -> stateless variant: { file_path, question } -> single-shot, no session
POST   /report                -> { session_id, dataset_name?, email_to? } -> deterministic report
GET    /sessions               -> list all persisted sessions (past + current chats)
GET    /sessions/{session_id}   -> one session's datasets + full chat history (for resuming)
DELETE /sessions/{session_id}    -> delete a session and its history
```

`/report` is deliberately LLM-free: report generation is a
deterministic pipeline (profile -> trends -> anomalies -> rankings ->
markdown), so it doesn't need to depend on the model correctly
routing tool calls to be reliable.

Run it directly:

```bash
python -m uvicorn app.api.main:app --reload --port 8000
```

Interactive docs: `http://localhost:8000/docs`

---

## 6. Reliability & safety

- Every tool wraps `pd.read_csv` and its core logic in `try/except`,
  returning `{"error": "..."}` instead of raising — including several
  pre-existing tools (`get_dataset_info`, `get_dataset_statistics`,
  `analyze_column`, `compare_categories`, `analyze_churn`,
  `create_churn_plot`) that were missing this and got fixed as part
  of this upgrade.
- The agent loop catches Ollama connection failures and returns a
  clear "model unavailable" answer instead of crashing the request.
- A `MAX_LLM_CALLS` cap (8) stops a runaway tool-calling loop.
- Unknown tool names, wrong tool arguments, and tool exceptions are
  all caught per-call and fed back to the model as an error result
  rather than crashing the turn.
- SQL is validated before execution (see above); no destructive
  statement ever reaches the database.
- `generate_business_report` / `deliver_business_report` never invent
  findings — every bullet is built from a real tool result, and a
  dataset with no usable date/metric/category columns still returns a
  clean (if minimal) report rather than fabricating one.
- `deliver_business_report` only reports `action: "report_saved"`
  after checking the files exist on disk, and only reports
  `email.sent: true` after `smtplib` completes without raising.
- Heavy optional dependencies (`faiss`, `sentence-transformers`) are
  imported lazily inside `semantic_search`, so the rest of the app
  (API, other tools, UI) still starts and works if they aren't
  installed.
- The zero-tool-call refusal guard in `agent.py` covers a *failed*
  tool call too, not just no call at all: if a JOIN's SQL errors out
  and the model then answers as if the tool weren't available, that
  answer is rejected and retried with the real DuckDB error fed back
  — not just when the model skips calling a tool entirely.
- `_call_ollama_chat` retries a transient network failure (e.g. a
  reset connection to a cloud-hosted Ollama model) a couple of times
  with backoff before surfacing an error, since a fresh connection
  after a mid-stream reset usually just works.
- `train_model` accepts an explicit `feature_columns` list, and even
  without one, silently drops any text column with more than 50
  unique values before one-hot encoding — a timestamp/id column left
  in by mistake used to make training hang for minutes instead of
  seconds.
- CSV columns are downcast from pandas' (>= 3.0) new `str` extension
  dtype back to plain `object` before being handed to DuckDB, which
  doesn't recognize that dtype yet — otherwise any query touching a
  text column (e.g. a JOIN or GROUP BY) fails.

---

## 7. Tech stack

- **Python 3.11**
- **Ollama** (`gpt-oss:20b-cloud`) — real tool-calling LLM
- **FastAPI** + **Pydantic** — API layer, request/response validation
- **Streamlit** — thin UI client
- **pandas / numpy / scikit-learn** — data analysis, anomaly detection, ML
- **DuckDB** — SQL over the uploaded CSV
- **FAISS + sentence-transformers** — semantic search
- **matplotlib** — visualizations
- **fpdf2 / python-docx** — PDF and Word report export (pure Python, no system dependencies)
- **pytest** — test suite

---

## 8. Installation

```bash
git clone <this repo>
cd AI-Data-Analyst-Agent
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # edit if needed (SMTP is optional)
```

You'll also need [Ollama](https://ollama.com) running and the model
pulled/available (`gpt-oss:20b-cloud` is an Ollama cloud model —
requires the `ollama` CLI signed in to a plan that supports cloud
models, or swap `AGENT_MODEL` in `.env` for a local model that
supports tool calling, e.g. `qwen2.5:7b`).

### Run locally (two processes)

```bash
# Terminal 1: API
python -m uvicorn app.api.main:app --reload --port 8000

# Terminal 2: UI
streamlit run app.py
```

Open `http://localhost:8501`.

### Run with Docker Compose

```bash
docker compose up --build
```

This starts the API (port 8000) and the UI (port 8501), sharing
`data/uploads/` and `outputs/` via bind mounts. Set `OLLAMA_HOST` in
`.env` if Ollama isn't reachable at
`http://host.docker.internal:11434`.

---

## 9. Environment variables

See `.env.example` for the full list with defaults. Key ones:

| Variable | Purpose |
|---|---|
| `AGENT_MODEL` | Ollama model used for tool calling |
| `API_BASE_URL` | Where the Streamlit UI finds the FastAPI backend |
| `MAX_HISTORY_MESSAGES` | How many conversation turns to keep per session |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` / `SMTP_FROM` | Optional — enables email report delivery |

---

## 10. Testing

```bash
pytest tests/ -v
```

55+ tests covering:
- CSV validation and dataset loading/profiling, including missing
  files and empty files
- SQL validation (accepts safe SELECTs, rejects DROP/DELETE/UPDATE/
  INSERT/ALTER/multi-statement/ATTACH) and execution
- Anomaly detection (univariate + multivariate + small-data fallback)
- Trend analysis and period comparison (including per-group ranking)
- `rank_categories`, `generate_business_report`,
  `deliver_business_report` (including "never fakes email success",
  and that generated PDF/DOCX files actually exist on disk)
- The agent tool-calling loop, with the LLM mocked so tests don't
  depend on Ollama being reachable: direct answers, real tool
  execution, unknown-tool handling, LLM-failure handling, the
  max-iterations safety cap, conversation-history forwarding, and a
  regression test for a failed JOIN query being retried with the
  real DuckDB error instead of returning a fabricated "tool
  unavailable" excuse
- FastAPI endpoints, including an end-to-end
  upload -> `/report` (deterministic) and
  upload -> `/chat` -> tool call -> answer (LLM mocked) scenario

The original manual smoke script (`test_agent.py` at the repo root)
is preserved for a quick real-Ollama sanity check.

---

## 11. Example prompts

```
Give me an overview of this dataset.
Why did revenue decrease last month?
Which products are underperforming?
Find unusual changes in revenue.
Which region has the highest growth?
What were the top 5 products by revenue?          (-> SQL)
Show me the revenue trend.                          (-> chart)
Compare this month's performance with last month.
Analyze customer churn and identify the main factors.
Create a business report from this dataset.
Generate and save a business report, and email it to ops@company.com.
```

---

## 12. Project structure

```
AI-Data-Analyst-Agent/
|-- app.py                     # Streamlit UI (thin API client)
|-- app/
|   |-- config.py              # env-driven settings
|   |-- agent/
|   |   |-- agent.py           # orchestrator: system prompt, tool schemas, loop
|   |   `-- memory.py          # in-memory session store
|   |-- api/
|   |   |-- main.py            # FastAPI app
|   |   `-- schemas.py         # Pydantic request/response models
|   |-- tools/
|   |   |-- data_tools.py
|   |   |-- analysis_tools.py
|   |   |-- sql_tools.py
|   |   |-- anomaly_tools.py
|   |   |-- trend_tools.py
|   |   |-- visualization_tools.py
|   |   |-- ml_tools.py
|   |   |-- semantic_search_tools.py
|   |   |-- report_tools.py
|   |   `-- automation_tools.py
|   `-- utils/
|       `-- validation.py
|-- tests/
|   |-- conftest.py
|   |-- test_data_tools.py
|   |-- test_sql_tools.py
|   |-- test_anomaly_tools.py
|   |-- test_trend_tools.py
|   |-- test_business_report.py
|   |-- test_agent.py
|   `-- test_api.py
|-- Dockerfile.api
|-- Dockerfile.ui
|-- docker-compose.yml
|-- requirements.txt
|-- .env.example
`-- test_agent.py              # original manual smoke script (preserved)
```

---

## 13. Limitations

- Session memory is in-process only (not persistent, not multi-worker
  safe) — documented tradeoff for a single-process portfolio deploy.
- Column auto-detection (date/metric/category) is name- and
  dtype-based heuristics; it can pick the wrong column on unusual
  schemas, in which case pass the column names explicitly.
- The SQL keyword blocklist can reject a legitimate query that merely
  contains a blocked word inside a string literal (fail-safe, not a
  bug).
- Correlation, feature importance, and anomaly flags reflect
  association within the dataset, not proven causation.
- Ollama's small context window (`num_ctx=4096`) caps how much
  conversation history and tool output the model can reason over in
  one turn; `MAX_HISTORY_MESSAGES` trims accordingly.
- Email delivery requires the person deploying it to supply SMTP
  credentials; there's no built-in email provider.

## 14. Future improvements

- Persist sessions in Redis/a database instead of in-process memory.
- Ask-for-clarification flow when a business term (e.g. "growth")
  doesn't map cleanly to a column.
- Streaming responses from `/chat` for perceived latency.
- Pluggable LLM backend (OpenAI/Anthropic) behind the same tool
  schemas, selectable via `AGENT_MODEL`/a provider flag.
- Scheduled/recurring report delivery (e.g. a weekly cron calling
  `/report`).

---

## 15. How this demonstrates the target skill set

| Skill | Where |
|---|---|
| **LLMs / Generative AI** | Ollama tool-calling agent, `think=False` reasoning-suppression for latency, structured tool schemas |
| **AI Agents / Agentic workflows** | `app/agent/agent.py`'s tool-calling loop: intent -> tool selection -> real execution -> reasoning over results -> possible follow-up call -> final answer |
| **Tool Calling** | 17 real Python functions exposed as Ollama tools, executed for real (never faked), results fed back into the conversation |
| **AI Automation** | `deliver_business_report`: a verified save-to-disk action plus optional real email delivery |
| **Data Analysis** | `analysis_tools.py`, `anomaly_tools.py`, `trend_tools.py`, `ml_tools.py` |
| **SQL** | `sql_tools.py` — DuckDB, validated read-only queries |
| **APIs** | `app/api/main.py` — FastAPI, Pydantic models, proper HTTP status codes |
| **FastAPI** | Full backend with `/health /upload /chat /analyze /report`, tested via `TestClient` |
| **Structured outputs** | API response schemas + orchestration-derived `tools_used`/`tool_log`, not LLM-guessed JSON |
| **Error handling** | Try/except around every I/O boundary, LLM-failure handling, SQL validation, iteration caps, never-fake-success automation |
| **Python** | The whole thing |