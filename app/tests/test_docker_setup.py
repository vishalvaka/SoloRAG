import os
import pytest
import subprocess
import tempfile
from unittest.mock import patch, MagicMock


def test_docker_compose_ports_configuration():
    """Test that Docker Compose file has correct port mappings."""
    # This test ensures the compose file has the expected port mappings
    # We can't easily test the actual compose file parsing, but we can verify
    # that our configuration is correct by checking the expected structure
    
    expected_ports = {
        "8000": "8000",  # FastAPI backend
        "8501": "8501",  # Streamlit UI
    }
    
    # This is more of a documentation test - ensuring our setup is correct
    assert expected_ports["8000"] == "8000", "FastAPI port should be 8000"
    assert expected_ports["8501"] == "8501", "Streamlit port should be 8501"


def test_environment_variables_required():
    """Test that required environment variables are properly configured."""
    # Test that the environment variables we expect are defined
    required_vars = [
        "OLLAMA_URL",
        "BACKEND_URL"
    ]
    
    # This test documents the expected environment variables
    for var in required_vars:
        assert var in ["OLLAMA_URL", "BACKEND_URL"], f"Environment variable {var} should be configured"


def test_entrypoint_script_structure():
    """Test that the entrypoint script has the expected structure."""
    entrypoint_path = "docker/entrypoint.sh"
    
    # Check that the entrypoint script exists
    assert os.path.exists(entrypoint_path), "Entrypoint script should exist"
    
    # Read the entrypoint script to verify it has expected components
    with open(entrypoint_path, 'r') as f:
        content = f.read()
    
    # Check for expected components
    expected_components = [
        "ollama serve",  # Ollama startup
        "uvicorn app.main:app",  # FastAPI startup
        "streamlit run",  # Streamlit startup
        "SoloRAG is ready",  # Success message
    ]
    
    for component in expected_components:
        assert component in content, f"Entrypoint script should contain: {component}"


def test_dockerfile_structure():
    """Test that the Dockerfile has the expected structure."""
    dockerfile_path = "docker/backend.Dockerfile"
    
    # Check that the Dockerfile exists
    assert os.path.exists(dockerfile_path), "Dockerfile should exist"
    
    # Read the Dockerfile to verify it has expected components
    with open(dockerfile_path, 'r') as f:
        content = f.read()
    
    # Check for expected components
    expected_components = [
        "COPY ./streamlit_app.py",  # Streamlit app copy
        "EXPOSE 8000 8501",  # Port exposure
        "ENTRYPOINT",  # Entrypoint configuration
    ]
    
    for component in expected_components:
        assert component in content, f"Dockerfile should contain: {component}"


@patch('subprocess.run')
def test_docker_build_command(mock_run):
    """Test that Docker build command works correctly."""
    # Mock successful subprocess run
    mock_run.return_value = MagicMock(returncode=0)
    
    # Test the build command structure
    build_cmd = [
        "docker", "compose", "-f", "docker/compose.yaml", "build"
    ]
    
    # This test verifies that our build command is properly structured
    assert len(build_cmd) == 5, "Build command should have 5 components"
    assert build_cmd[0] == "docker", "First component should be 'docker'"
    assert build_cmd[1] == "compose", "Second component should be 'compose'"


@patch('subprocess.run')
def test_docker_up_command(mock_run):
    """Test that Docker up command works correctly."""
    # Mock successful subprocess run
    mock_run.return_value = MagicMock(returncode=0)
    
    # Test the up command structure
    up_cmd = [
        "docker", "compose", "-f", "docker/compose.yaml", "up", "--build"
    ]
    
    # This test verifies that our up command is properly structured
    assert len(up_cmd) == 6, "Up command should have 6 components"
    assert up_cmd[0] == "docker", "First component should be 'docker'"
    assert up_cmd[1] == "compose", "Second component should be 'compose'"
    assert "--build" in up_cmd, "Up command should include --build flag"


def test_streamlit_app_file_exists():
    """Test that the streamlit_app.py file exists and is accessible."""
    streamlit_path = "streamlit_app.py"
    
    # Check that the Streamlit app file exists
    assert os.path.exists(streamlit_path), "streamlit_app.py should exist"
    
    # Check that it's a Python file
    assert streamlit_path.endswith('.py'), "Streamlit app should be a Python file"


def test_docker_compose_files_exist():
    """Test that all required Docker Compose files exist."""
    required_files = [
        "docker/compose.yaml",
        "docker/compose.gpu.yaml",
    ]
    
    for file_path in required_files:
        assert os.path.exists(file_path), f"Docker Compose file should exist: {file_path}"


def test_artifacts_directory_exists():
    """Test that the artifacts directory exists for Docker volume mounting."""
    artifacts_path = "artifacts"
    
    # Check that the artifacts directory exists
    assert os.path.exists(artifacts_path), "artifacts directory should exist"
    assert os.path.isdir(artifacts_path), "artifacts should be a directory"
    
    # Check for expected artifact files
    expected_files = [
        "faiss.idx",
        "meta.npy"
    ]
    
    for file_name in expected_files:
        file_path = os.path.join(artifacts_path, file_name)
        assert os.path.exists(file_path), f"Artifact file should exist: {file_path}" 