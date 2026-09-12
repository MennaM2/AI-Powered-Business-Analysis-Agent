from typing import Optional

from pydantic import BaseModel, Field


class DatasetMeta(BaseModel):
    name: str
    table_name: str
    rows: int
    columns: int
    missing_values: int


class UploadResponse(BaseModel):
    session_id: str
    dataset: DatasetMeta
    # Every table now available in this session, including ones
    # uploaded in earlier calls - confirms nothing was wiped.
    all_datasets: list[str]


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1)


class ToolLogEntry(BaseModel):
    tool: str
    arguments: dict
    success: bool


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    tools_used: list[str] = []
    tool_log: list[ToolLogEntry] = []
    plot_path: Optional[str] = None
    latency_seconds: Optional[float] = None
    error: Optional[str] = None
    error_detail: Optional[str] = None
    report_files: Optional[dict] = None


class AnalyzeRequest(BaseModel):
    session_id: Optional[str] = None
    file_path: Optional[str] = None
    question: str = Field(..., min_length=1)


class ReportRequest(BaseModel):
    session_id: Optional[str] = None
    # Which uploaded table to report on. Optional only when the
    # session has exactly one dataset.
    dataset_name: Optional[str] = None
    file_path: Optional[str] = None
    date_column: Optional[str] = None
    value_column: Optional[str] = None
    category_column: Optional[str] = None
    target_column: Optional[str] = None
    email_to: Optional[str] = None


class ReportResponse(BaseModel):
    status: str
    report_path_markdown: Optional[str] = None
    report_path_html: Optional[str] = None
    detected_columns: Optional[dict] = None
    key_insights: list[str] = []
    recommendations: list[str] = []
    email: Optional[dict] = None
    error: Optional[str] = None
    report_path_pdf: Optional[str] = None
    report_path_docx: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    model: str


class SessionSummary(BaseModel):
    session_id: str
    title: Optional[str] = None
    created_at: float
    message_count: int
    datasets: list[str] = []


class SessionMessage(BaseModel):
    role: str
    content: str
    created_at: float


class SessionDetail(BaseModel):
    session_id: str
    title: Optional[str] = None
    created_at: float
    datasets: dict[str, str]
    history: list[SessionMessage]