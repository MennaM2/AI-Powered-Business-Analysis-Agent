"""
Centralized, environment-driven configuration.

Nothing here has a hidden default that matters for correctness - every
value can be overridden via an environment variable (see .env.example)
so the same code runs the same way locally, in Docker, or in CI.
"""

import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # LLM
    model: str = os.getenv("AGENT_MODEL", "gpt-oss:20b-cloud")

    # Storage
    upload_dir: str = os.getenv("UPLOAD_DIR", "data/uploads")
    output_dir: str = os.getenv("OUTPUT_DIR", "outputs")

    # Persistent session store (SQLite). Sessions, their datasets, and
    # their full chat history live here - they survive an API restart
    # and are never wiped by uploading another file.
    sessions_db_path: str = os.getenv("SESSIONS_DB_PATH", "data/sessions.db")

    # Conversation memory
    max_history_messages: int = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))

    # Automation: email delivery (optional)
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_from: str = os.getenv("SMTP_FROM", "") or os.getenv("SMTP_USER", "")
    smtp_use_tls: bool = _get_bool("SMTP_USE_TLS", True)

    # API
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")

    # Per-call timeout (seconds) for each individual Ollama chat()
    # call. Without this, a single stalled call blocks the whole
    # request indefinitely instead of failing with a clear error.
    ollama_timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))

    @property
    def smtp_configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_user and self.smtp_password)


settings = Settings()