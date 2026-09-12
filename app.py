# python -m streamlit run app.py
#
# This UI is a thin client: it never calls the agent directly. Every
# analysis question, upload, and report request goes over HTTP to the
# FastAPI backend (app/api/main.py), which is the real "production"
# surface of this project. Run the API first:
#   python -m uvicorn app.api.main:app --port 8000

import io
import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


st.set_page_config(
    page_title="AI Business Analysis Agent",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"], [data-testid="stSidebar"] {
    font-family: Inter, "Segoe UI", sans-serif;
}

.stApp {
    background: #F4F6F8;
}

[data-testid="stHeader"] {
    background: transparent;
    height: 3rem;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stDecoration"] {display: none;}
.stDeployButton {display: none;}

[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #475569;
}

.brand-mark {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}

.brand-mark .mark {
    width: 28px;
    height: 28px;
    border-radius: 7px;
    background: #1B3A4B;
    color: #F8FAFC;
    font-size: 13px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    letter-spacing: 0.02em;
}

.brand-mark .name {
    font-size: 15px;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.02em;
}

.sidebar-kicker {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748B;
    margin: 22px 0 10px 0;
}

.app-header {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
}

.app-header h1 {
    margin: 0;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #0F172A;
}

.app-header p {
    margin: 6px 0 0 0;
    font-size: 13.5px;
    color: #64748B;
    line-height: 1.5;
    max-width: 640px;
}

.status-pill {
    flex-shrink: 0;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 12px;
    border-radius: 999px;
    border: 1px solid #E2E8F0;
    background: #F8FAFC;
    color: #475569;
    white-space: nowrap;
}

.status-pill.ready {
    background: #ECFDF3;
    border-color: #BBF7D0;
    color: #166534;
}

.status-pill.error {
    background: #FEF2F2;
    border-color: #FECACA;
    color: #991B1B;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
}

.metric-card .metric-label {
    font-size: 11px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}

.metric-card .metric-value {
    font-size: 20px;
    font-weight: 700;
    color: #1B3A4B;
    letter-spacing: -0.03em;
}

.file-chip {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 13px;
    color: #0F172A;
    word-break: break-all;
}

.empty-state {
    background: #FFFFFF;
    border: 1px dashed #CBD5E1;
    border-radius: 14px;
    padding: 48px 32px;
    text-align: center;
}

.empty-state h2 {
    margin: 0 0 8px 0;
    font-size: 18px;
    font-weight: 650;
    color: #0F172A;
}

.empty-state p {
    margin: 0 auto;
    max-width: 460px;
    color: #64748B;
    font-size: 14px;
    line-height: 1.6;
}

.section-title {
    font-size: 13px;
    font-weight: 650;
    color: #0F172A;
    letter-spacing: -0.01em;
    margin: 0 0 10px 0;
}

.panel {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px 18px 8px 18px;
    margin-bottom: 16px;
}

.viz-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 12px;
    margin-top: 8px;
}

.meta-line {
    font-size: 12px;
    color: #64748B;
}

.tools-used {
    font-size: 11.5px;
    color: #475569;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 999px;
    padding: 3px 10px;
    display: inline-block;
    margin: 2px 4px 2px 0;
}

[data-testid="stChatMessage"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 8px 6px;
    margin-bottom: 8px;
    font-size: 13px;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
    font-size: 13px !important;
    line-height: 1.55;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #F1F5F4;
    border-color: #D9E2E1;
}

[data-testid="stChatInput"] {
    border-top: 1px solid #E2E8F0;
    padding-top: 8px;
}

.stButton > button {
    background: #1B3A4B;
    color: #FFFFFF;
    border: 1px solid #1B3A4B;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
}

.stButton > button:hover {
    background: #16303E;
    border-color: #16303E;
    color: #FFFFFF;
}

[data-testid="stFileUploader"] section {
    background: #F8FAFC;
    border: 1px dashed #CBD5E1;
    border-radius: 10px;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0;
    border-radius: 10px;
}

div[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
}

h2, h3 {
    color: #0F172A !important;
    font-weight: 650 !important;
    letter-spacing: -0.02em;
}

[data-testid="stCaptionContainer"] {
    color: #64748B;
}

.chat-history-item {
    display: block;
    width: 100%;
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 2px;
}

.chat-history-item:hover {
    background: #F8FAFC;
}

.chat-history-item.active {
    background: #EEF2FF;
    border-color: #C7D2FE;
}

.chat-history-title {
    font-size: 13px;
    font-weight: 600;
    color: #0F172A;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.chat-history-meta {
    font-size: 11px;
    color: #94A3B8;
    margin-top: 1px;
}
</style>
""",
    unsafe_allow_html=True,
)

def _api_url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}{path}"


def _reset_local_session():
    """Start a fresh chat in the UI. This never deletes anything on
    the server - past sessions stay listed under 'Resume a chat'."""
    st.session_state.session_id = None
    st.session_state.datasets = []
    st.session_state.preview_frames = {}
    st.session_state.messages = []
    st.session_state.uploaded_signatures = set()


if "session_id" not in st.session_state:
    _reset_local_session()

if "api_reachable" not in st.session_state:
    st.session_state.api_reachable = True


with st.sidebar:
    st.markdown(
        """
        <div class="brand-mark">
            <div class="mark">AA</div>
            <div class="name">Business Analysis Agent</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Backend: {API_BASE_URL}")

    if st.button("+ New chat", use_container_width=True):
        _reset_local_session()
        st.rerun()

    st.markdown('<p class="sidebar-kicker">Datasets</p>', unsafe_allow_html=True)
    st.caption(
        "Upload one or more CSVs. Uploading another file adds a "
        "table to this chat - it never replaces what's already here."
    )

    uploaded_files = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or more tabular CSV files.",
        label_visibility="collapsed",
    )

    upload_error = None

    for uploaded_file in uploaded_files or []:
        file_signature = (uploaded_file.name, uploaded_file.size)
        if file_signature in st.session_state.uploaded_signatures:
            continue

        file_bytes = uploaded_file.getvalue()

        try:
            params = (
                {"session_id": st.session_state.session_id}
                if st.session_state.session_id
                else None
            )
            response = requests.post(
                _api_url("/upload"),
                files={"file": (uploaded_file.name, file_bytes, "text/csv")},
                params=params,
                timeout=30,
            )
            st.session_state.api_reachable = True
        except requests.exceptions.RequestException as exc:
            upload_error = (
                f"Could not reach the analysis API at {API_BASE_URL}. "
                f"Is it running? ({exc})"
            )
            st.session_state.api_reachable = False
            response = None

        if response is not None:
            if response.status_code == 200:
                payload = response.json()
                st.session_state.session_id = payload["session_id"]
                st.session_state.datasets = payload["all_datasets"]
                st.session_state.uploaded_signatures.add(file_signature)

                table_name = payload["dataset"]["table_name"]
                try:
                    st.session_state.preview_frames[table_name] = pd.read_csv(
                        io.BytesIO(file_bytes)
                    )
                except Exception:
                    pass
            else:
                try:
                    upload_error = response.json().get("detail", "Upload failed.")
                except ValueError:
                    upload_error = "Upload failed."

    if st.session_state.datasets:
        st.markdown('<p class="sidebar-kicker">Loaded tables</p>', unsafe_allow_html=True)
        for table_name in st.session_state.datasets:
            st.markdown(
                f'<div class="file-chip">{table_name}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No datasets loaded yet.")

    st.markdown('<p class="sidebar-kicker">Resume a chat</p>', unsafe_allow_html=True)
    try:
        sessions_response = requests.get(_api_url("/sessions"), timeout=10)
        past_sessions = sessions_response.json() if sessions_response.status_code == 200 else []
    except requests.exceptions.RequestException:
        past_sessions = []

    def _load_session(session_id: str) -> None:
        """Switch the UI to an existing chat - same data this used
        to load via the dropdown, just triggered by clicking the
        chat's own row instead of picking it then pressing a
        separate 'Load' button."""
        try:
            detail_response = requests.get(
                _api_url(f"/sessions/{session_id}"), timeout=15
            )
            detail_response.raise_for_status()
            detail = detail_response.json()

            st.session_state.session_id = detail["session_id"]
            st.session_state.datasets = list(detail["datasets"].keys())
            st.session_state.preview_frames = {}
            st.session_state.uploaded_signatures = set()
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]}
                for m in detail["history"]
            ]
            st.rerun()
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not load that chat: {exc}")

    if past_sessions:
        for past_session in past_sessions:
            is_active = past_session["session_id"] == st.session_state.session_id

            # Title comes from the backend: the dataset name at
            # upload time, replaced by the first question once one's
            # been asked (see app/api/main.py). Only ever falls back
            # to raw dataset names here if an older chat predates
            # that titling logic and has no title saved.
            title = past_session.get("title") or (
                ", ".join(past_session["datasets"]) or "Untitled chat"
            )
            meta = (
                f"{', '.join(past_session['datasets']) or 'no datasets'} "
                f"\u00b7 {past_session['message_count']} messages"
            )

            st.button(
                title,
                key=f"chat_item_{past_session['session_id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                help=meta,
                disabled=is_active,
                on_click=_load_session,
                args=(past_session["session_id"],),
            )
    else:
        st.caption("No saved chats yet.")

    st.markdown('<p class="sidebar-kicker">Example questions</p>', unsafe_allow_html=True)
    st.caption(
        "Why did revenue decrease last month?  \n"
        "Which products are underperforming?  \n"
        "Using SQL, join orders with users by country.  \n"
        "Create a business report from this dataset."
    )

    if st.session_state.datasets:
        st.markdown('<p class="sidebar-kicker">Automation</p>', unsafe_allow_html=True)
        with st.form("report_form", clear_on_submit=False):
            report_dataset = (
                st.session_state.datasets[0]
                if len(st.session_state.datasets) == 1
                else st.selectbox("Dataset for report", st.session_state.datasets)
            )
            email_to = st.text_input(
                "Email the report to (optional)",
                placeholder="name@company.com",
                label_visibility="collapsed",
            )
            generate_clicked = st.form_submit_button(
                "Generate & save business report", use_container_width=True
            )

        if generate_clicked:
            with st.spinner("Generating business report…"):
                try:
                    response = requests.post(
                        _api_url("/report"),
                        json={
                            "session_id": st.session_state.session_id,
                            "dataset_name": report_dataset,
                            "email_to": email_to or None,
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    report_result = response.json()
                except requests.exceptions.RequestException as exc:
                    st.error(f"Could not reach the API: {exc}")
                    report_result = None

                if report_result:
                    if report_result.get("status") == "success":
                        st.success(
                            f"Report saved: "
                            f"{report_result['report_path_markdown']}"
                        )

                        download_cols = st.columns(4)
                        download_specs = [
                            ("report_path_pdf", "PDF", "application/pdf"),
                            (
                                "report_path_docx",
                                "Word",
                                "application/vnd.openxmlformats-"
                                "officedocument.wordprocessingml.document"
                            ),
                            ("report_path_html", "HTML", "text/html"),
                            ("report_path_markdown", "Markdown", "text/markdown"),
                        ]
                        for col, (key, label, mime) in zip(
                            download_cols, download_specs
                        ):
                            file_path = report_result.get(key)
                            if file_path and os.path.exists(file_path):
                                with open(file_path, "rb") as report_file:
                                    col.download_button(
                                        f"Download {label}",
                                        data=report_file.read(),
                                        file_name=os.path.basename(file_path),
                                        mime=mime,
                                        use_container_width=True,
                                        key=f"download_{key}",
                                    )

                        email_status = report_result.get("email")
                        if email_status:
                            if email_status.get("sent"):
                                st.caption(
                                    f"Emailed to {email_status.get('to')} "
                                    f"(PDF attached)."
                                )
                            elif email_status.get("attempted"):
                                st.caption(
                                    f"Email attempted but failed: "
                                    f"{email_status.get('error')}"
                                )
                            else:
                                st.caption(email_status.get("reason", ""))
                    else:
                        st.error(
                            report_result.get("error", "Report generation failed.")
                        )

    if st.session_state.messages:
        if st.button("Clear this chat's messages", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


if not st.session_state.api_reachable:
    status_label = "API unreachable"
    status_class = "status-pill error"
elif st.session_state.datasets:
    status_label = f"{len(st.session_state.datasets)} dataset(s) ready"
    status_class = "status-pill ready"
else:
    status_label = "Awaiting dataset"
    status_class = "status-pill"

st.markdown(
    f"""
    <div class="app-header">
        <div>
            <h1>AI-Powered Business Analysis & Automation Agent</h1>
            <p>Ask business questions in natural language. Upload multiple
            related tables and the agent can join across them with SQL,
            run analysis/trend/anomaly tools on any one of them, and
            generate an automated report - all with a chat history that's
            saved and resumable.</p>
        </div>
        <div class="{status_class}">{status_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if upload_error:
    st.error(upload_error)

if st.session_state.preview_frames:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-title">Dataset preview</p>',
        unsafe_allow_html=True,
    )
    preview_tabs = st.tabs(list(st.session_state.preview_frames.keys()))
    for tab, (table_name, df) in zip(
        preview_tabs, st.session_state.preview_frames.items()
    ):
        with tab:
            st.caption(
                f"Showing the first 10 of {len(df):,} rows · "
                f"{len(df.columns)} columns"
            )
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


if st.session_state.session_id and st.session_state.datasets:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-title">Conversation</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.caption(
            "Ask about trends, anomalies, top/bottom performers, SQL "
            "queries (including joins across your tables), or request "
            "a full business report."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("tools_used"):
                st.markdown(
                    "".join(
                        f'<span class="tools-used">{tool}</span>'
                        for tool in message["tools_used"]
                    ),
                    unsafe_allow_html=True,
                )

    question = st.chat_input("Ask a business question about these datasets…")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your dataset…"):
                try:
                    response = requests.post(
                        _api_url("/chat"),
                        json={
                            "session_id": st.session_state.session_id,
                            "message": question,
                        },
                        timeout=300,
                    )
                    response.raise_for_status()
                    result = response.json()

                    answer = result.get("answer") or (
                        "The agent didn't return an answer."
                    )
                    st.write(answer)

                    tools_used = result.get("tools_used", [])
                    if tools_used:
                        st.markdown(
                            "".join(
                                f'<span class="tools-used">{tool}</span>'
                                for tool in tools_used
                            ),
                            unsafe_allow_html=True,
                        )

                    if result.get("latency_seconds") is not None:
                        st.caption(
                            f"Agent latency: {result['latency_seconds']}s "
                            f"| Tool calls: {len(result.get('tool_log', []))}"
                        )

                    if result.get("error_detail"):
                        with st.expander("Error details"):
                            st.code(result["error_detail"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "tools_used": tools_used,
                    })

                    plot_path = result.get("plot_path")

                    if plot_path and os.path.exists(plot_path):
                        st.markdown(
                            '<div class="viz-card">',
                            unsafe_allow_html=True,
                        )
                        st.image(
                            plot_path,
                            caption="Generated visualization",
                            use_container_width=True,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                    report_files = result.get("report_files")
                    if report_files:
                        # A report tool was called from this chat
                        # question (not just the dedicated Report
                        # Generation form) - offer the same download
                        # buttons here so the files are actually
                        # reachable, not just named in the answer text.
                        download_specs = [
                            ("pdf", "PDF", "application/pdf"),
                            (
                                "docx", "Word",
                                "application/vnd.openxmlformats-"
                                "officedocument.wordprocessingml.document"
                            ),
                            ("html", "HTML", "text/html"),
                            ("markdown", "Markdown", "text/markdown"),
                        ]
                        available = [
                            (key, label, mime)
                            for key, label, mime in download_specs
                            if report_files.get(key)
                            and os.path.exists(report_files[key])
                        ]
                        if available:
                            download_cols = st.columns(len(available))
                            for col, (key, label, mime) in zip(
                                download_cols, available
                            ):
                                file_path = report_files[key]
                                with open(file_path, "rb") as report_file:
                                    col.download_button(
                                        f"Download {label}",
                                        data=report_file.read(),
                                        file_name=os.path.basename(file_path),
                                        mime=mime,
                                        use_container_width=True,
                                        key=f"chat_download_{key}_"
                                            f"{len(st.session_state.messages)}",
                                    )

                except requests.exceptions.RequestException as exc:
                    st.error(
                        f"Could not reach the analysis API at "
                        f"{API_BASE_URL}. Is it running? ({exc})"
                    )
                except Exception as exc:
                    print(f"[Agent Error] {exc}")
                    st.error(
                        "Sorry, I couldn't analyze that request. "
                        "Please try rephrasing your question or check "
                        "that the dataset contains the data needed to "
                        "answer it."
                    )

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class="empty-state">
            <h2>Upload a dataset to begin</h2>
            <p>Use the sidebar to add one or more CSV files, or resume a
            previously saved chat. After a dataset loads, you can inspect
            a preview, ask the agent business questions (including ones
            that join across multiple tables), or generate an automated
            report.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )