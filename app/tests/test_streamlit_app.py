import importlib
import os
import pytest
import sys
from unittest.mock import patch, AsyncMock

streamlit = pytest.importorskip("streamlit", reason="streamlit not installed")


def test_streamlit_app_imports():
    """Streamlit script should import without side effects/errors."""
    try:
        module = importlib.import_module("streamlit_app")
    except (KeyError, AttributeError):
        pytest.skip("Session state not initialised outside Streamlit runtime")
    assert module.QUERY_ENDPOINT.endswith("/query/stream")


def test_backend_url_environment_variable():
    """Test that BACKEND_URL environment variable is properly configured."""
    # Test default value
    with patch.dict(os.environ, {}, clear=True):
        import streamlit_app
        assert streamlit_app.BACKEND_URL == "http://localhost:8000"
    
    # Test custom value
    with patch.dict(os.environ, {"BACKEND_URL": "http://custom-backend:8000"}, clear=True):
        import importlib
        importlib.reload(streamlit_app)
        assert streamlit_app.BACKEND_URL == "http://custom-backend:8000"


def test_query_endpoint_construction():
    """Test that QUERY_ENDPOINT is properly constructed from BACKEND_URL."""
    with patch.dict(os.environ, {"BACKEND_URL": "http://test-backend:8000"}, clear=True):
        import importlib
        import streamlit_app
        importlib.reload(streamlit_app)
        assert streamlit_app.QUERY_ENDPOINT == "http://test-backend:8000/query/stream"


@pytest.mark.asyncio
async def test_fetch_stream_function():
    """Test the fetch_stream async function with mocked HTTP client."""
    with patch.dict(os.environ, {"BACKEND_URL": "http://localhost:8000"}, clear=True):
        import importlib
        import streamlit_app
        importlib.reload(streamlit_app)
        
        # Mock the httpx.AsyncClient
        with patch('streamlit_app.httpx.AsyncClient') as mock_client:
            # Setup mock response
            mock_response = AsyncMock()
            mock_response.aiter_text.return_value = iter([
                "This is a test answer",
                "[SOURCES]",
                '[{"text": "Source 1", "score": 0.95}]'
            ])
            
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_instance.stream.return_value.__aenter__.return_value = mock_response
            mock_client.return_value = mock_client_instance
            
            # Test the fetch_stream function
            question = "What is Stripe?"
            body, sources = await streamlit_app.fetch_stream(question)
            
            # Verify the response
            assert "This is a test answer" in body
            assert "Source 1" in sources
            assert "0.95" in sources
            
            # Verify the HTTP call was made correctly
            mock_client_instance.stream.assert_called_once_with(
                "POST", 
                "http://localhost:8000/query/stream", 
                json={"question": question}
            )


def test_streamlit_app_configuration():
    """Test that Streamlit app is properly configured."""
    with patch.dict(os.environ, {"BACKEND_URL": "http://localhost:8000"}, clear=True):
        import importlib
        import streamlit_app
        importlib.reload(streamlit_app)
        
        # Test that the app has the expected configuration
        assert hasattr(streamlit_app, 'st')
        assert hasattr(streamlit_app, 'BACKEND_URL')
        assert hasattr(streamlit_app, 'QUERY_ENDPOINT') 