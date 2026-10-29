"""
Structured JSON logging for Angeles City Garbage Route Optimizer.
Thesis AL-1: Audit trail for optimization runs, API calls, and user interactions.
"""
import logging
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines for structured analysis."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_entry["data"] = getattr(record, "extra_data")

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_logger(
    name: str = "angeles_optimizer",
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up a structured JSON logger.

    Args:
        name: Logger name (default: "angeles_optimizer")
        log_dir: Directory for log files (default: ./logs)
        level: Logging level (default: INFO)
        console_output: Whether to also output to console (default: True)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Set default log directory
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)

    # File handler with JSON formatting
    log_file = log_dir / f"optimizer_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    # Console handler for development (human-readable)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    return logger


def log_optimization_run(
    logger: logging.Logger,
    params: Dict[str, Any],
    results: Dict[str, Any],
    runtime_seconds: float,
) -> None:
    """
    Log a complete optimization run with parameters and results.

    Args:
        logger: Logger instance
        params: GA parameters used
        results: Optimization results (routes, distances, etc.)
        runtime_seconds: Execution time in seconds
    """
    extra = {
        "extra_data": {
            "event_type": "optimization_run",
            "params": {
                "n_vehicles": params.get("n_vehicles"),
                "pop_size": params.get("pop_size"),
                "generations": params.get("generations"),
                "mutation_rate": params.get("mutation_rate"),
                "crossover_rate": params.get("crossover_rate"),
                "random_seed": params.get("random_seed"),
                "selected_barangays": params.get("selected", []),
            },
            "results": {
                "total_distance_km": results.get("total"),
                "num_routes": len(results.get("routes", [])),
                "avg_distance_per_vehicle": results.get("total", 0)
                / max(len(results.get("routes", [])), 1),
            },
            "runtime_seconds": round(runtime_seconds, 2),
        }
    }
    logger.info("Optimization run completed", extra=extra)


def log_api_call(
    logger: logging.Logger,
    api_name: str,
    endpoint: str,
    success: bool,
    error: Optional[str] = None,
    response_time_ms: Optional[float] = None,
) -> None:
    """
    Log an external API call (e.g., ORS routing).

    Args:
        logger: Logger instance
        api_name: API name (e.g., "ORS")
        endpoint: API endpoint called
        success: Whether the call succeeded
        error: Error message if failed
        response_time_ms: Response time in milliseconds
    """
    extra = {
        "extra_data": {
            "event_type": "api_call",
            "api_name": api_name,
            "endpoint": endpoint,
            "success": success,
            "response_time_ms": response_time_ms,
        }
    }

    if success:
        logger.info(f"{api_name} API call successful", extra=extra)
    else:
        extra["extra_data"]["error"] = error
        logger.warning(f"{api_name} API call failed: {error}", extra=extra)


def log_user_action(
    logger: logging.Logger,
    action: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a user interaction for UX research.

    Args:
        logger: Logger instance
        action: Action name (e.g., "run_optimization", "export_data")
        details: Additional context about the action
    """
    extra = {
        "extra_data": {
            "event_type": "user_action",
            "action": action,
            "details": details or {},
        }
    }
    logger.info(f"User action: {action}", extra=extra)


def log_error(
    logger: logging.Logger,
    error: Exception,
    context: Optional[str] = None,
) -> None:
    """
    Log an error with full stack trace.

    Args:
        logger: Logger instance
        error: Exception object
        context: Additional context about where the error occurred
    """
    extra = {
        "extra_data": {
            "event_type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
        }
    }
    logger.error(f"Error in {context or 'unknown context'}: {error}", extra=extra, exc_info=True)


# Default logger instance for easy import
logger = setup_logger()
