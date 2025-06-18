# SoloRAG – A Self-Hosted, Offline-First RAG Demo

SoloRAG is a minimal, self-contained Retrieval-Augmented-Generation stack that lets you chat with your documents **entirely offline**. It's designed for easy setup and experimentation, supporting both CPU and GPU environments.

*   **Backend**: Python with FastAPI
*   **Retrieval**: Sentence-Transformers and a FAISS vector index
*   **LLM**: Ollama-hosted model (defaults to `llama3:8b-instruct-q5_K_M`)
*   **Deployment**: A single Docker container for the entire stack.

> 🔎 **Looking for the design rationale?** See [`docs/architecture.md`](docs/architecture.md).

> 📚 **Documentation**
> * **Quick local setup** → [`docs/setup_local.md`](docs/setup_local.md)
> * **Architecture overview** → [`docs/architecture.md`](docs/architecture.md)
> * **Benchmarks & metrics** → [`docs/benchmarks.md`](docs/benchmarks.md)

---

## 🚀 Quick Start (Recommended)

The easiest way to get started is with Docker Compose. This method packages the entire application—the FastAPI backend and the Ollama LLM—into a single container.

**Prerequisites**:
*   [Docker](https://docs.docker.com/engine/install/) installed.
*   For GPU support, the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is required.

### 1. Clone the Repository
```bash
git clone https://github.com/vishalvaka/SoloRAG.git
cd SoloRAG
```

### 2. Run with Docker Compose

You can launch the application in either CPU or GPU mode.

#### **CPU Mode** (Default)
This will run on any machine with Docker installed.
```bash
docker compose -f docker/compose.yaml up --build
```
> **Note**: The first build might take a while as it downloads the base images and PyTorch binaries. Subsequent builds will be much faster. For detailed build logs, use `docker compose -f docker/compose.yaml build --progress=plain`.

#### **GPU Mode**
This leverages your NVIDIA GPU for significantly faster inference.
```bash
docker compose -f docker/compose.yaml -f docker/compose.gpu.yaml up --build
```
This command layers the GPU-specific configuration over the base setup, ensuring the correct CUDA environment and dependencies are used.

### 3. Customizing the LLM
By default, the application uses the `llama3:8b-instruct-q5_K_M` model. You can switch to any other model from the [Ollama Library](https://ollama.com/library) by setting the `OLLAMA_MODEL` environment variable.

Open `docker/compose.yaml` and modify the `environment` section for the `backend` service:
```yaml
# docker/compose.yaml
services:
  backend:
    ...
    environment:
      - OLLAMA_URL=http://localhost:11434
      # Uncomment and set the model you want to use.
      - OLLAMA_MODEL=mistral:7b-instruct-q5_K_M
```
When you next run `docker compose -f docker/compose.yaml up`, the entrypoint script will automatically pull the specified model.

### 4. Query the API
Once the container is running and you see the log `Uvicorn running on http://0.0.0.0:8000`, open a new terminal and send a request:
```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "When is my first payout on Stripe?"}' | jq
```

---

## 💻 Local Development Setup (Without Docker)

If you prefer to run the application directly on your machine, follow these steps.

### 1. Environment Setup
Create and activate a Python virtual environment.
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
The project uses separate requirements files for CPU and GPU to manage dependencies like PyTorch and FAISS.

First, install the common packages:
```bash
pip install -r requirements/common.txt
```

Next, install the packages for your target hardware:
```bash
# For a CPU-only setup
pip install -r requirements/cpu.txt

# For a GPU setup (requires a CUDA toolkit on your system)
# pip install -r requirements/gpu.txt
```

Finally, install `sentence-transformers` (which depends on PyTorch) and the development dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Run Ollama
The application requires a running Ollama instance.

First, install and start the Ollama service by following their official [installation guide](https://ollama.com/).

Then, pull the default model:
```bash
ollama pull llama3:8b-instruct-q5_K_M
```

### 4. Run the Application
Set the `OLLAMA_URL` environment variable and start the FastAPI server:
```bash
export OLLAMA_URL=http://localhost:11434
uvicorn app.main:app --reload
```
The API will be available at `http://localhost:8000`.

---

## 🧪 Testing Guide

The project includes a comprehensive test suite to ensure correctness and stability.

### Running All Tests
A convenience script is provided to run the entire test suite using `pytest`.
```bash
./scripts/run_tests.sh
```
You can also pass `pytest` arguments directly to the script:
```bash
# Run tests with verbose output and print statements
./scripts/run_tests.sh -v -s

# Run only tests containing "api" in their name
./scripts/run_tests.sh -k "api"
```

### Test Suite Overview
The test suite is located in `app/tests/` and covers the following components:

*   `test_api.py`: Tests the FastAPI endpoints. It ensures that the `/query` endpoint handles valid requests correctly, rejects invalid ones (e.g., empty questions), and returns the expected JSON structure.
*   `test_retrieval.py`: Unit tests the retrieval logic. It verifies that the retriever correctly finds relevant document chunks from the FAISS index and handles cases with no matching context.
*   `test_prompt.py`: Unit tests the prompt generation logic. It checks that the final prompt sent to the LLM is correctly formatted based on the retrieved context.
*   `test_ollama_client.py`: Unit tests the asynchronous Ollama client. It mocks the `httpx` library to ensure the client correctly sends requests, handles successful responses, and manages retries upon failure.

---

## 🛠️ Advanced Usage

### Manual Docker Builds
If you need more control, you can build the Docker images manually without Compose.

**Build CPU Image:**
```bash
docker build -t solorag-backend-cpu \
  --build-arg BUILD_MODE=cpu \
  -f docker/backend.Dockerfile .
```

**Build GPU Image:**
```bash
docker build -t solorag-backend-gpu \
  --build-arg BUILD_MODE=gpu \
  --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
  -f docker/backend.Dockerfile .
```

### Building the Vector Index
The repository includes a pre-built FAISS index in `artifacts/`. To rebuild it from a raw data file (e.g., `data/raw/stripe_faqs_full.jsonl`), run the `build_index.py` script:
```bash
# Ensure you have an appropriate model from sentence-transformers
# The default is "all-MiniLM-L6-v2"
python scripts/build_index.py
```

---

## 📂 Project Structure

```
SoloRAG/
├── README.md
├── app/
│   ├── main.py               # FastAPI routes and application logic
│   ├── retrieval.py          # Core RAG retrieval functions
│   ├── prompt.py             # Prompt formatting logic
│   ├── ollama_client.py      # Asynchronous client for Ollama
│   └── tests/                # Pytest suite
│
├── artifacts/
│   ├── faiss.idx             # Pre-built FAISS vector index
│   └── meta.npy              # Metadata for the index
│
├── data/
│   ├── raw/                  # Raw source documents (JSONL format)
│   └── eval/                 # Evaluation datasets
│
├── docker/
│   ├── backend.Dockerfile    # Dockerfile for both CPU and GPU builds
│   ├── compose.yaml          # Base Docker Compose for CPU
│   ├── compose.gpu.yaml      # Docker Compose override for GPU
│   └── entrypoint.sh         # Container startup script
│
├── requirements/
│   ├── common.txt            # Common Python packages
│   ├── cpu.txt               # CPU-specific deps (faiss-cpu, torch+cpu)
│   └── gpu.txt               # GPU-specific deps (faiss-gpu, torch+cuda)
│
├── scripts/
│   ├── build_index.py        # Script to build the FAISS index
│   ├── evaluate.py           # Script to run evaluations
│   └── run_tests.sh          # Test runner script
│
└── ...
```

---

## 📜 License

SoloRAG is released under the MIT License. See [`LICENSE`](LICENSE) for details.