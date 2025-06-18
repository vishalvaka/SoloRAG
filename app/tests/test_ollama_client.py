# app/tests/test_ollama_client.py
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from app.ollama_client import generate, OLLAMA_URL, OLLAMA_MODEL


@pytest.mark.asyncio
async def test_generate_happy_path():
    """Successful API call returns the generated content."""
    mock_response_dict = {"response": "Here is the generated text."}
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value=mock_response_dict)
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    # Async context manager that yields mock_client
    mock_acm = AsyncMock()
    mock_acm.__aenter__.return_value = mock_client
    mock_acm.__aexit__.return_value = MagicMock()

    with patch("httpx.AsyncClient", return_value=mock_acm):
        result = await generate("test prompt")
        assert result == "Here is the generated text."
        mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_generate_retry_logic():
    """Client should retry on 503 Service Unavailable errors."""
    mock_unavailable = MagicMock(spec=httpx.Response)
    mock_unavailable.status_code = 503
    mock_unavailable.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
        "Service Unavailable", request=AsyncMock(), response=mock_unavailable
    ))

    mock_success_response = {"response": "Success after retry"}
    mock_success = MagicMock(spec=httpx.Response)
    mock_success.status_code = 200
    mock_success.json = MagicMock(return_value=mock_success_response)
    mock_success.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=[mock_unavailable, mock_unavailable, mock_success])

    mock_acm = AsyncMock()
    mock_acm.__aenter__.return_value = mock_client
    mock_acm.__aexit__.return_value = MagicMock()

    with patch("httpx.AsyncClient", return_value=mock_acm):
        result = await generate("test prompt", retries=3, delay_s=0.01)
        assert result == "Success after retry"
        assert mock_client.post.call_count == 3


@pytest.mark.asyncio
async def test_generate_all_retries_fail():
    """If all retries fail, the exception should be raised."""
    def _fail_response():
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 500
        resp.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
            "Server Error", request=AsyncMock(), response=resp
        ))
        return resp

    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=lambda *a, **kw: _fail_response())

    mock_acm = AsyncMock()
    mock_acm.__aenter__.return_value = mock_client
    mock_acm.__aexit__.return_value = MagicMock()

    with patch("httpx.AsyncClient", return_value=mock_acm):
        with pytest.raises(httpx.HTTPStatusError):
            await generate("test prompt", retries=2, delay_s=0.01)
        assert mock_client.post.call_count == 2