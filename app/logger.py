import logging, sys

try:
    import structlog  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    structlog = None  # type: ignore


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

    def _configure_logging():
        """Configure structlog for JSON output compatible with Grafana Loki."""
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=logging.INFO,
        )

        structlog.configure(  # type: ignore[attr-defined]
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),  # type: ignore[attr-defined]
                structlog.processors.add_log_level,  # type: ignore[attr-defined]
                structlog.processors.StackInfoRenderer(),  # type: ignore[attr-defined]
                structlog.processors.format_exc_info,  # type: ignore[attr-defined]
                structlog.processors.JSONRenderer(),  # type: ignore[attr-defined]
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),  # type: ignore[attr-defined]
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),  # type: ignore[attr-defined]
            cache_logger_on_first_use=True,
        )

    _configure_logging()
    logger = structlog.get_logger()  # type: ignore[attr-defined] 