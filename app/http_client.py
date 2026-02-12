# app/http_client.py
"""Proxy-aware httpx client factory.

All outbound HTTP in the application should use ``get_http_client()``
so that corporate proxy and custom CA certificate settings are honored
automatically.
"""

from __future__ import annotations

import ssl
from typing import Optional

import httpx

from .config import get_settings


def _build_ssl_context() -> Optional[ssl.SSLContext]:
    """Return an SSL context that trusts the custom CA bundle, if configured."""
    settings = get_settings()
    if not settings.CUSTOM_CA_BUNDLE:
        return None
    ctx = ssl.create_default_context(cafile=settings.CUSTOM_CA_BUNDLE)
    return ctx


def _build_proxy_mounts() -> Optional[dict[str, httpx.AsyncHTTPTransport]]:
    """Build httpx proxy mounts from env-configured proxy settings."""
    settings = get_settings()
    proxies: dict[str, httpx.AsyncHTTPTransport] = {}
    ssl_ctx = _build_ssl_context()

    if settings.HTTP_PROXY:
        proxies["http://"] = httpx.AsyncHTTPTransport(
            proxy=settings.HTTP_PROXY,
        )
    if settings.HTTPS_PROXY:
        kwargs = {"proxy": settings.HTTPS_PROXY}
        if ssl_ctx:
            kwargs["verify"] = ssl_ctx  # type: ignore[assignment]
        proxies["https://"] = httpx.AsyncHTTPTransport(**kwargs)  # type: ignore[arg-type]

    return proxies if proxies else None


def get_http_client(timeout: Optional[float] = 30.0, **kwargs) -> httpx.AsyncClient:
    """Return a configured ``httpx.AsyncClient``.

    - Honors ``HTTP_PROXY`` / ``HTTPS_PROXY`` / ``NO_PROXY``
    - Loads custom CA bundle from ``CUSTOM_CA_BUNDLE``
    - Pass additional kwargs through to ``httpx.AsyncClient``.
    """
    settings = get_settings()
    ssl_ctx = _build_ssl_context()
    mounts = _build_proxy_mounts()

    client_kwargs: dict = {"timeout": timeout, **kwargs}
    if ssl_ctx and "verify" not in client_kwargs:
        client_kwargs["verify"] = ssl_ctx
    if mounts:
        client_kwargs["mounts"] = mounts

    return httpx.AsyncClient(**client_kwargs)
