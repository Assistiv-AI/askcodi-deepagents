"""Unit tests for the web_fetch tool."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from deepagents.middleware.web_fetch import create_web_fetch_tool


def _html_response(url: str, body: str) -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("GET", url),
        headers={"content-type": "text/html; charset=utf-8"},
        text=body,
    )


def _text_response(url: str, body: str) -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("GET", url),
        headers={"content-type": "text/plain"},
        text=body,
    )


def test_successful_fetch() -> None:
    tool = create_web_fetch_tool()
    fake_resp = _text_response("https://example.com", "hello world")

    with patch("deepagents.middleware.web_fetch.httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value.__enter__.return_value
        mock_client.get.return_value = fake_resp

        result = tool.func(urls=["https://example.com"])

    assert "hello world" in result
    assert "--- https://example.com ---" in result


def test_html_tags_stripped() -> None:
    tool = create_web_fetch_tool()
    fake_resp = _html_response("https://example.com", "<h1>Title</h1><p>Body</p>")

    with patch("deepagents.middleware.web_fetch.httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value.__enter__.return_value
        mock_client.get.return_value = fake_resp

        result = tool.func(urls=["https://example.com"])

    assert "<h1>" not in result
    assert "Title" in result
    assert "Body" in result


def test_multiple_urls() -> None:
    tool = create_web_fetch_tool()
    responses = {
        "https://a.com": _text_response("https://a.com", "content a"),
        "https://b.com": _text_response("https://b.com", "content b"),
    }

    with patch("deepagents.middleware.web_fetch.httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value.__enter__.return_value
        mock_client.get.side_effect = lambda url: responses[url]

        result = tool.func(urls=["https://a.com", "https://b.com"])

    assert "content a" in result
    assert "content b" in result
    assert "--- https://a.com ---" in result
    assert "--- https://b.com ---" in result


def test_domain_restriction_allowed() -> None:
    tool = create_web_fetch_tool(allowed_domains=["example.com"])
    fake_resp = _text_response("https://example.com/page", "ok")

    with patch("deepagents.middleware.web_fetch.httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value.__enter__.return_value
        mock_client.get.return_value = fake_resp

        result = tool.func(urls=["https://example.com/page"])

    assert "ok" in result


def test_domain_restriction_subdomain_allowed() -> None:
    tool = create_web_fetch_tool(allowed_domains=["example.com"])
    fake_resp = _text_response("https://sub.example.com/page", "sub ok")

    with patch("deepagents.middleware.web_fetch.httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value.__enter__.return_value
        mock_client.get.return_value = fake_resp

        result = tool.func(urls=["https://sub.example.com/page"])

    assert "sub ok" in result


def test_domain_restriction_blocked() -> None:
    tool = create_web_fetch_tool(allowed_domains=["example.com"])
    result = tool.func(urls=["https://evil.com/steal"])

    assert "Error" in result
    assert "Domain not allowed" in result


def test_invalid_url() -> None:
    tool = create_web_fetch_tool()
    result = tool.func(urls=["not-a-url"])

    assert "Error" in result
    assert "Invalid URL" in result


def test_unsupported_scheme() -> None:
    tool = create_web_fetch_tool()
    result = tool.func(urls=["ftp://example.com/file"])

    assert "Error" in result
    assert "Unsupported scheme" in result


def test_http_error_handled() -> None:
    tool = create_web_fetch_tool()

    with patch("deepagents.middleware.web_fetch.httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value.__enter__.return_value
        mock_client.get.side_effect = httpx.ConnectError("connection refused")

        result = tool.func(urls=["https://example.com"])

    assert "Error" in result


@pytest.mark.asyncio
async def test_async_fetch() -> None:
    tool = create_web_fetch_tool()
    fake_resp = _text_response("https://example.com", "async hello")

    with patch("deepagents.middleware.web_fetch.httpx.AsyncClient") as mock_cls:
        mock_client = mock_cls.return_value.__aenter__ = AsyncMock(
            return_value=AsyncMock()
        )
        mock_instance = await mock_client()
        mock_instance.get = AsyncMock(return_value=fake_resp)
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await tool.coroutine(urls=["https://example.com"])

    assert "async hello" in result


def test_schema_is_json_serializable() -> None:
    tool = create_web_fetch_tool()
    schema = tool.tool_call_schema.model_json_schema()
    properties = schema.get("properties", {})

    assert "urls" in properties


def test_schema_with_allowed_domains() -> None:
    tool = create_web_fetch_tool(allowed_domains=["example.com"])
    schema = tool.tool_call_schema.model_json_schema()
    properties = schema.get("properties", {})

    assert set(properties) == {"urls"}
