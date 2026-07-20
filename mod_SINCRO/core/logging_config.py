"""Configuración de logging estructurado para GammaSync."""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
from datetime import datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formateador JSON para logs estructurados."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Agregar contexto extra si existe
        if hasattr(record, "context"):
            log_entry["context"] = record.context

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class GammaSyncLogger:
    """Logger centralizado para GammaSync."""

    _instance: GammaSyncLogger | None = None
    _logger: logging.Logger | None = None

    def __new__(cls) -> GammaSyncLogger:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._logger is None:
            self._setup_logger()

    def _setup_logger(self):
        """Configura el logger con handlers para consola y archivo."""
        self._logger = logging.getLogger("gammasync")
        self._logger.setLevel(logging.DEBUG)

        # Evitar duplicados
        if self._logger.handlers:
            return

        # Formato para consola (legible)
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%H:%M:%S",
        )

        # Handler consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        self._logger.addHandler(console_handler)

        # Handler archivo (JSON estructurado)
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"gammasync_{datetime.now().strftime('%Y%m%d')}.log")

        # Rotación por tamaño (10MB) y backups (5)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        self._logger.addHandler(file_handler)

        self._logger.info("Logger GammaSync inicializado", extra={"context": {"log_dir": log_dir}})

    def get_logger(self) -> logging.Logger:
        """Retorna el logger configurado."""
        return self._logger

    def log_with_context(
        self,
        level: int,
        message: str,
        context: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Log con contexto adicional."""
        extra = {"context": context} if context else {}
        extra.update(kwargs)
        self._logger.log(level, message, extra=extra)

    def debug(self, message: str, context: dict[str, Any] | None = None):
        self.log_with_context(logging.DEBUG, message, context)

    def info(self, message: str, context: dict[str, Any] | None = None):
        self.log_with_context(logging.INFO, message, context)

    def warning(self, message: str, context: dict[str, Any] | None = None):
        self.log_with_context(logging.WARNING, message, context)

    def error(self, message: str, context: dict[str, Any] | None = None):
        self.log_with_context(logging.ERROR, message, context)

    def log_processing_start(self, study_path: str, params: dict):
        """Log inicio de procesamiento."""
        self.info("Procesamiento iniciado", context={
            "study_path": study_path,
            "processing_params": params,
        })

    def log_processing_end(self, study_path: str, duration_sec: float, metrics: dict):
        """Log fin de procesamiento."""
        self.info("Procesamiento completado", context={
            "study_path": study_path,
            "duration_sec": round(duration_sec, 2),
            "metrics_summary": {
                "phase_sd": metrics.get("phase_sd"),
                "bandwidth": metrics.get("bandwidth"),
                "technical_classification": metrics.get("technical_classification"),
            },
        })

    def log_segmentation(self, method: str, n_voxels: int, n_slices: int):
        """Log segmentación."""
        self.info("Segmentación completada", context={
            "method": method,
            "n_voxels": n_voxels,
            "n_slices": n_slices,
        })

    def log_export(self, export_type: str, output_path: str):
        """Log exportación."""
        self.info("Exportación generada", context={
            "export_type": export_type,
            "output_path": output_path,
        })

    def log_error(self, error: Exception, context: dict[str, Any] | None = None):
        """Log error con excepción."""
        ctx = context or {}
        ctx["error_type"] = type(error).__name__
        ctx["error_message"] = str(error)
        self.error(f"Error: {error}", context=ctx)


# Instancia global
_logger_instance: GammaSyncLogger | None = None


def get_logger() -> GammaSyncLogger:
    """Retorna la instancia global del logger."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = GammaSyncLogger()
    return _logger_instance


def setup_logging(level: str = "INFO"):
    """Configura el nivel de logging global."""
    logger = get_logger().get_logger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Actualizar handler de consola
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.handlers.RotatingFileHandler):
            handler.setLevel(numeric_level)
