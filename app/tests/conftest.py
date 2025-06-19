import sys, pathlib, pytest
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

# Avoid real HTTP calls to Ollama during tests
@pytest.fixture(autouse=True)
def _stub_ollama(monkeypatch):
    async def _fake_generate(prompt: str, *args, **kwargs):
        return "Test answer."

    from app import ollama_client as oc
    monkeypatch.setattr(oc, "generate", _fake_generate)
    monkeypatch.setattr(oc, "stream_generate", lambda prompt: (_ for _ in []))  # empty async gen

    from app import retrieval as retr
    monkeypatch.setattr(retr, "call_ollama", _fake_generate, raising=False)
    monkeypatch.setattr(retr, "call_ollama_stream", lambda prompt: (_ for _ in []), raising=False)

    yield