import logging, sys

try:
    import structlog
except ModuleNotFoundError:  # pragma: no cover
    structlog = None


if structlog is None:
    # Fallback to stdlib logging so the rest of the codebase keeps working.
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    class _CompatLogger:
        """Minimal shim so calls like logger.info("event", details="x") don't crash."""

        def __init__(self, inner: logging.Logger):
            self._inner = inner

        def info(self, event: str, **kwargs):
            message = f"{event} | {kwargs}" if kwargs else event
            self._inner.info(message)

        def warning(self, event: str, **kwargs):
            message = f"{event} | {kwargs}" if kwargs else event
            self._inner.warning(message)

        def error(self, event: str, **kwargs):
            message = f"{event} | {kwargs}" if kwargs else event
            self._inner.error(message)

        def debug(self, event: str, **kwargs):
            message = f"{event} | {kwargs}" if kwargs else event
            self._inner.debug(message)

        def bind(self, **kwargs):
            # No-op for compatibility with structlog's bind()
            return self

    logger = _CompatLogger(logging.getLogger("solorag"))
else:

    def _configure_logging() -> None:
        """Configure structlog for JSON output compatible with Grafana Loki."""
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=logging.INFO,
        )

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    _configure_logging()
    logger = structlog.get_logger() 