"""Web fetch tool for retrieving content from public URLs.

This module provides a ``web_fetch`` tool that agents can use to fetch content
from one or more URLs concurrently.  It is additive and does not modify any
existing middleware.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from typing import Annotated
from urllib.parse import urlparse

import httpx
from langchain_core.tools import BaseTool, StructuredTool

_DEFAULT_TIMEOUT = 30
_MAX_RESPONSE_BYTES = 1_048_576  # 1 MB per URL

_TAG_RE = re.compile(r"<[^>]+>")

WEB_FETCH_TOOL_DESCRIPTION = """Fetches content from one or more public URLs.

Use this tool to retrieve web page content or API responses.

Usage:
- Pass a list of URLs to fetch.
- HTML responses are returned as plain text with tags stripped.
- Non-HTML responses are returned as-is (up to 1 MB per URL).
- Each result is labelled with its source URL.
"""


def _strip_html(text: str) -> str:
    """Remove HTML tags from *text* and collapse whitespace."""
    return _TAG_RE.sub("", text)


def _validate_domain(url: str, allowed_domains: Sequence[str] | None) -> None:
    """Raise ``ValueError`` if *url* is not on an allowed domain."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        msg = f"Invalid URL: {url}"
        raise ValueError(msg)
    if parsed.scheme not in ("http", "https"):
        msg = f"Unsupported scheme: {parsed.scheme}"
        raise ValueError(msg)
    if allowed_domains is not None:
        hostname = parsed.hostname or ""
        if not any(hostname == d or hostname.endswith(f".{d}") for d in allowed_domains):
            msg = f"Domain not allowed: {hostname}"
            raise ValueError(msg)


def _format_response(url: str, response: httpx.Response) -> str:
    """Format a single HTTP response for return to the agent."""
    content_type = response.headers.get("content-type", "")
    body = response.text[:_MAX_RESPONSE_BYTES]
    if "html" in content_type:
        body = _strip_html(body)
    return f"--- {url} ---\n{body}"


def _format_error(url: str, error: Exception) -> str:
    return f"--- {url} ---\nError: {error}"


def create_web_fetch_tool(
    *,
    allowed_domains: Sequence[str] | None = None,
) -> BaseTool:
    """Create a ``web_fetch`` tool for fetching content from URLs.

    Args:
        allowed_domains: If provided, only URLs matching these domains are
            permitted.  Subdomains are matched automatically (e.g. ``"example.com"``
            allows ``"sub.example.com"``).

    Returns:
        A ``BaseTool`` instance.
    """

    def sync_web_fetch(
        urls: Annotated[list[str], "List of URLs to fetch."],
    ) -> str:
        """Synchronous web fetch implementation."""
        results: list[str] = []
        for url in urls:
            try:
                _validate_domain(url, allowed_domains)
            except ValueError as exc:
                results.append(_format_error(url, exc))
                continue
            try:
                with httpx.Client(timeout=_DEFAULT_TIMEOUT, follow_redirects=True) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                results.append(_format_response(url, resp))
            except Exception as exc:  # noqa: BLE001
                results.append(_format_error(url, exc))
        return "\n\n".join(results)

    async def async_web_fetch(
        urls: Annotated[list[str], "List of URLs to fetch."],
    ) -> str:
        """Asynchronous web fetch with concurrent requests."""

        async def _fetch_one(client: httpx.AsyncClient, url: str) -> str:
            try:
                _validate_domain(url, allowed_domains)
            except ValueError as exc:
                return _format_error(url, exc)
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                return _format_response(url, resp)
            except Exception as exc:  # noqa: BLE001
                return _format_error(url, exc)

        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT, follow_redirects=True) as client:
            results = await asyncio.gather(*[_fetch_one(client, url) for url in urls])
        return "\n\n".join(results)

    return StructuredTool.from_function(
        name="web_fetch",
        description=WEB_FETCH_TOOL_DESCRIPTION,
        func=sync_web_fetch,
        coroutine=async_web_fetch,
    )


__all__ = ["create_web_fetch_tool", "WEB_FETCH_TOOL_DESCRIPTION"]
