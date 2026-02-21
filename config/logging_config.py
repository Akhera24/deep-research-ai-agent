"""
Comprehensive Logging System for Research Agent

Dual-purpose logging system:
1. Structlog for application logging (development/production)
2. JSONL for execution logs (evaluation/audit trail)


Features:
- Structured logging (structlog)
- JSONL execution logs (JSON Lines format)
- Dual output (console + file)
- Event tracking (search, extraction, risk, etc.)
- Performance metrics
- Cost tracking
- Production-ready
- Consistent log format
- Comprehensive event tracking
- Machine-parseable output
- Thread-safe operations
"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from contextlib import contextmanager
import structlog

from config.settings import settings


# ============================================================================
# PART 1: STRUCTLOG CONFIGURATION (for application logging)
# ============================================================================

def configure_structlog():
    """
    Configure structlog for application logging.
    
    Development: Pretty console output with colors
    Production: JSON output for log aggregation
    
    This is called automatically on import.
    """
    # Shared processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    
    # Environment-specific rendering
    if settings.ENVIRONMENT == "development":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.LOG_LEVEL)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    THIS IS THE MAIN FUNCTION FOR APPLICATION LOGGING.
    Use this throughout the codebase for general logging.
    
    Args:
        name: Logger name (typically __name__ of calling module)
              If None, returns root logger
    
    Returns:
        Configured structlog logger
        
    Example:
        >>> from config.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started", user_id=123)
    """
    if name:
        return structlog.get_logger(name)
    else:
        return structlog.get_logger()


# ============================================================================
# PART 2: JSONL EXECUTION LOGGING (for evaluation/audit)
# ============================================================================

class JSONLFormatter(logging.Formatter):
    """
    Format log records as JSON Lines (JSONL).
    
    Each log record becomes a single-line JSON object.
    Perfect for append-only logging and stream processing.
    
    Example Output:
    {"timestamp": "2026-01-07T01:23:45Z", "level": "INFO", "event": "search_executed", "data": {...}}
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Convert log record to JSONL format."""
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
        }
        
        # Add message if present
        if record.getMessage():
            log_data["message"] = record.getMessage()
        
        # Add custom fields from extra parameter
        if hasattr(record, 'run_id'):
            log_data['run_id'] = record.run_id
        if hasattr(record, 'event_type'):
            log_data['event_type'] = record.event_type
        if hasattr(record, 'event_data'):
            log_data['data'] = record.event_data
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Return single-line JSON
        return json.dumps(log_data, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable formatter for console output.
    
    Makes logs easy to read during development and debugging.
    """
    
    # ANSI color codes for pretty output
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format record with colors and structure"""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Basic format
        timestamp = datetime.utcnow().strftime('%H:%M:%S')
        level = f"{color}{record.levelname:8s}{reset}"
        message = record.getMessage()
        
        # Add event type if present
        if hasattr(record, 'event_type'):
            message = f"[{record.event_type}] {message}"
        
        return f"{timestamp} {level} {message}"


def setup_execution_logging(
    run_id: str,
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Set up JSONL execution logging for a research run.
    
    THIS IS FOR EXECUTION LOGS (PDF REQUIREMENT).
    Creates dual output:
    1. Console: Pretty, colored, human-readable
    2. File: JSONL format, machine-readable
    
    Args:
        run_id: Unique identifier for this research run
        log_dir: Directory for log files
        console_level: Minimum level for console
        file_level: Minimum level for file
    
    Returns:
        Configured logger for execution logging
    
    Example:
        >>> exec_logger = setup_execution_logging("run_abc123")
        >>> log_event(exec_logger, "search_executed", "run_abc123", {"query": "Sarah Chen"})
    
    Output File (logs/run_abc123.jsonl):
        {"timestamp": "2026-01-07T01:23:45Z", "event_type": "search_executed", "data": {...}}
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f"execution.{run_id}")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Handler 1: Console (pretty)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ConsoleFormatter())
    logger.addHandler(console_handler)
    
    # Handler 2: File (JSONL)
    log_file = log_path / f"{run_id}.jsonl"
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(JSONLFormatter())
    logger.addHandler(file_handler)
    
    # Prevent propagation
    logger.propagate = False
    
    logger.info(f"Execution logging initialized for run {run_id}")
    logger.debug(f"Log file: {log_file}")
    
    return logger


def log_event(
    logger: logging.Logger,
    event_type: str,
    run_id: str,
    event_data: Dict[str, Any],
    level: int = logging.INFO
) -> None:
    """
    Log a structured event.
    
    PRIMARY way to log execution events.
    Every significant action should call this.
    
    Args:
        logger: Logger from setup_execution_logging()
        event_type: Event type (e.g., "search_executed")
        run_id: Research run identifier
        event_data: Event-specific data
        level: Logging level
    
    Example:
        >>> log_event(exec_logger, "search_executed", "run_123", {
        ...     "query": "Sarah Chen",
        ...     "results_count": 10
        ... })
    """
    extra = {
        'event_type': event_type,
        'run_id': run_id,
        'event_data': event_data
    }
    
    # Create message for console
    message = f"{event_type}"
    if 'query' in event_data:
        message += f": {event_data['query']}"
    
    logger.log(level, message, extra=extra)


@contextmanager
def log_stage(logger: logging.Logger, stage_name: str, run_id: str):
    """
    Context manager for logging stage entry/exit with timing.
    
    Args:
        logger: Execution logger
        stage_name: Stage name
        run_id: Research run identifier
    
    Example:
        >>> with log_stage(exec_logger, "data_collection", "run_123"):
        ...     collect_data()
    """
    start_time = datetime.utcnow()
    
    log_event(logger, "stage_started", run_id, {
        "stage": stage_name,
        "start_time": start_time.isoformat() + "Z"
    })
    
    try:
        yield
    except Exception as e:
        duration = (datetime.utcnow() - start_time).total_seconds()
        log_event(logger, "stage_failed", run_id, {
            "stage": stage_name,
            "duration_seconds": duration,
            "error": str(e),
            "error_type": type(e).__name__
        }, level=logging.ERROR)
        raise
    else:
        duration = (datetime.utcnow() - start_time).total_seconds()
        log_event(logger, "stage_completed", run_id, {
            "stage": stage_name,
            "duration_seconds": duration
        })


def log_model_call(
    logger: logging.Logger,
    run_id: str,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
    duration_seconds: float,
    success: bool = True
) -> None:
    """
    Log AI model API call with metrics.
    
    Critical for cost tracking and performance monitoring.
    """
    log_event(logger, "model_called", run_id, {
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": round(cost, 6),
        "duration_seconds": round(duration_seconds, 3),
        "success": success
    })


def log_search_execution(
    logger: logging.Logger,
    run_id: str,
    query: str,
    search_engine: str,
    results_count: int,
    duration_seconds: float
) -> None:
    """Log search query execution."""
    log_event(logger, "search_executed", run_id, {
        "query": query,
        "engine": search_engine,
        "results_count": results_count,
        "duration_seconds": round(duration_seconds, 3)
    })


def log_fact_extraction(
    logger: logging.Logger,
    run_id: str,
    fact_id: str,
    category: str,
    confidence: float,
    source_count: int
) -> None:
    """Log fact extraction event."""
    log_event(logger, "fact_extracted", run_id, {
        "fact_id": fact_id,
        "category": category,
        "confidence": round(confidence, 2),
        "source_count": source_count
    })


def log_risk_detection(
    logger: logging.Logger,
    run_id: str,
    risk_id: str,
    category: str,
    severity: str,
    confidence: float
) -> None:
    """Log risk flag detection."""
    log_event(logger, "risk_flagged", run_id, {
        "risk_id": risk_id,
        "category": category,
        "severity": severity,
        "confidence": round(confidence, 2)
    })


# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize structlog on import
configure_structlog()

# Module-level application logger
logger = get_logger(__name__)

# Log configuration
logger.info(
    "Logging system initialized",
    environment=settings.ENVIRONMENT,
    log_level=settings.LOG_LEVEL,
    features=["structlog", "jsonl_execution_logs"]
)


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Structlog (application logging)
    "get_logger",
    "logger",
    "configure_structlog",
    
    # JSONL execution logging (for evaluation)
    "setup_execution_logging",
    "log_event",
    "log_stage",
    "log_model_call",
    "log_search_execution",
    "log_fact_extraction",
    "log_risk_detection",
    
    # Formatters
    "JSONLFormatter",
    "ConsoleFormatter"
]