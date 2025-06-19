import importlib, pytest, sys

streamlit = pytest.importorskip("streamlit", reason="streamlit not installed")

def test_streamlit_app_imports():
    """Streamlit script should import without side effects/errors."""
    module = importlib.import_module("streamlit_app")
    assert module.QUERY_ENDPOINT.endswith("/query/stream") 