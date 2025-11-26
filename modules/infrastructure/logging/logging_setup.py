# modules/logging_setup.py
from __future__ import annotations

import json
import logging
import os
import re
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QCoreApplication, QTimer, QtMsgType, qInstallMessageHandler
from PySide6.QtWidgets import QApplication, QMessageBox

# ---------- context & redaction ----------

_SESSION_ID = f"session-{int(time.time())}"
try:
    from contextvars import ContextVar

    _corr_id: ContextVar[str | None] = ContextVar("corr_id", default=None)
except Exception:
    _corr_id = None  # Py3.7+ 應該都有, 保底


class ContextFilter(logging.Filter):
    """Filter to inject session and correlation IDs into log records."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = _SESSION_ID
        if _corr_id:
            cid = _corr_id.get()
        else:
            cid = None
        record.correlation_id = cid or "-"
        return True


# 例示性去識別化, 視需要增補
_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{2,4}[-.\s]?){2,4}\d\b")
_TW_UNI_RE = re.compile(r"\b[0-9]{8}\b")  # 簡單示意


class PiiRedactionFilter(logging.Filter):
    """Filter to redact PII (email, phone, ID) from log messages."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        msg = _EMAIL_RE.sub("[email]", msg)
        msg = _PHONE_RE.sub("[phone]", msg)
        msg = _TW_UNI_RE.sub("[id]", msg)
        record.msg = msg
        record.args = ()
        return True


# ---------- JSON formatter ----------


class JsonFormatter(logging.Formatter):
    """Formatter to output log records as JSON."""
    def format(self, record: logging.LogRecord) -> str:
        data: Dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
            "session": getattr(record, "session_id", "-"),
            "corr": getattr(record, "correlation_id", "-"),
        }
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


# ---------- UI bridge with rate limit & dedupe ----------


class _UiLogBridge:
    """Bridge to display log messages in the UI status bar or as popups."""
    def __init__(self) -> None:
        self._status = None
        self._parent = None
        self._last_popup_at = 0.0
        self._last_popup_msg = ""
        self.rate_limit_ms = 1500
        self.popup_level = logging.ERROR

    def bind(self, parent, status, rate_limit_ms: int, popup_level: int) -> None:
        self._parent = parent
        self._status = status
        self.rate_limit_ms = int(rate_limit_ms)
        self.popup_level = int(popup_level)

    def show_status(self, text: str, persistent: bool) -> None:
        if not self._status:
            return
        if persistent:
            self._status.message(text)
        else:
            self._status.message_temp(text, ms=2500)

    def show_popup(self, title: str, text: str) -> None:
        now = time.time() * 1000
        if now - self._last_popup_at < self.rate_limit_ms:
            return
        if text == self._last_popup_msg:
            return
        self._last_popup_at = now
        self._last_popup_msg = text
        if self._parent:
            QMessageBox.critical(self._parent, title, text)


class QtUiHandler(logging.Handler):
    """Logging handler that forwards messages to the Qt UI bridge."""
    def __init__(self, level=logging.INFO):
        super().__init__(level)
        self.setLevel(level)

    def emit(self, record: logging.LogRecord) -> None:
        app = QApplication.instance()
        if app is None:
            return
        bridge: Optional[_UiLogBridge] = app.property("ui_log_bridge")
        if not isinstance(bridge, _UiLogBridge):
            return
        # 透過 Qt 事件佇列進 UI 執行緒
        msg = self.format(record)

        def _do():
            if record.levelno >= logging.ERROR:
                bridge.show_status(f"錯誤: {msg}", persistent=True)
                if record.levelno >= bridge.popup_level:
                    bridge.show_popup("發生錯誤", msg)
            elif record.levelno >= logging.WARNING:
                bridge.show_status(f"警告: {msg}", persistent=False)
            else:
                bridge.show_status(msg, persistent=False)

        QTimer.singleShot(0, _do)


# ---------- setup & Qt message proxy ----------


def _ensure_bridge() -> _UiLogBridge | None:
    app = QApplication.instance()
    if app is None:
        return None
    b = app.property("ui_log_bridge")
    if not isinstance(b, _UiLogBridge):
        b = _UiLogBridge()
        app.setProperty("ui_log_bridge", b)
    return b


def install_ui_targets(parent, status, rate_limit_ms: int, popup_level: int) -> None:
    b = _ensure_bridge()
    if b:
        b.bind(parent, status, rate_limit_ms, popup_level)


def _env_level(default: int) -> int:
    val = os.getenv("APP_LOG_LEVEL", "")
    mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return mapping.get(val.upper(), default)


def setup_logging(
    level: int,
    log_dir: Path | str,
    json_enabled: bool,
    max_bytes: int,
    backup_count: int,
) -> None:
    """Initialize the logging configuration for the application.

    Sets up file handlers (text and JSON), console handler, and UI handler.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    text_path = log_dir / "app.log"
    json_path = log_dir / "app.jsonl"

    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(_env_level(level))

    ctx = ContextFilter()
    pii = PiiRedactionFilter()

    # text file
    fh = RotatingFileHandler(
        str(text_path), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s %(session_id)s %(correlation_id)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    fh.addFilter(ctx)
    fh.addFilter(pii)

    # console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    ch.addFilter(pii)

    root.addHandler(fh)
    root.addHandler(ch)

    # json file
    if json_enabled:
        jh = RotatingFileHandler(
            str(json_path), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        jh.setLevel(logging.DEBUG)
        jf = JsonFormatter()
        jh.setFormatter(jf)
        jh.addFilter(ctx)
        jh.addFilter(pii)
        root.addHandler(jh)

    # UI handler
    qh = QtUiHandler(level=logging.INFO)
    qh.setFormatter(logging.Formatter("%(message)s"))
    qh.addFilter(pii)
    root.addHandler(qh)


def install_qt_message_proxy() -> None:
    """Install a Qt message handler to redirect Qt messages to Python logging."""
    # 將 Qt 自身訊息導到 logging
    level_map = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def handler(mode, context, message):
        lvl = level_map.get(mode, logging.INFO)
        logging.getLogger("Qt").log(lvl, message)

    qInstallMessageHandler(handler)


# ---------- helper API ----------


def set_correlation_id(cid: str | None) -> None:
    """Set the correlation ID for the current context."""
    if _corr_id:
        _corr_id.set(cid)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name or __name__)
