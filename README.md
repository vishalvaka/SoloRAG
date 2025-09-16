# SoloRAG – A Self-Hosted, Offline-First RAG Demo

SoloRAG is a minimal, self-contained Retrieval-Augmented-Generation stack that lets you chat with your documents **entirely offline**. It's designed for easy setup and experimentation, supporting both CPU and GPU environments.

*   **Backend**: Python with FastAPI
*   **Retrieval**: Sentence-Transformers and a FAISS vector index (**CPU by default; optional GPU**)
*   **LLM**: Ollama-hosted model (defaults to `tinyllama:latest` for fast startup)
*   **Deployment**: A single Docker container for the entire stack.

> 🔎 **Looking for the design rationale?** See [`docs/architecture.md`](docs/architecture.md).

> 📚 **Documentation**
> * **Quick local setup** → [`docs/setup_local.md`](docs/setup_local.md)
> * **Architecture overview** → [`docs/architecture.md`](docs/architecture.md)
> * **Benchmarks & metrics** → [`docs/benchmarks.md`](docs/benchmarks.md)

---

## ⚡ GPU Acceleration (Optional)

SoloRAG supports **CUDA-optimized FAISS** for dramatically faster vector search when running with GPU support. The system can use the GPU if enabled, but for fastest startup and broad compatibility, retrieval runs on CPU by default.

> To enable GPU retrieval, set `FAISS_FORCE_CPU=0` in the backend environment (CPU is forced by default).

### 🚀 Performance Benefits

When using the GPU Docker configuration, you get:

* **10-20x faster vector search** compared to CPU
* **5-10x faster index building** with GPU-accelerated clustering  
* **Automatic optimization** with larger overfetch ratios for better accuracy
* **Memory efficiency** with FP16 optimization and smart batch sizing
* **Graceful fallback** to CPU if GPU operations fail

### 🎯 Index Types

* **CPU Mode (default)**: `IndexFlatIP` for exact similarity search
* **GPU Mode (opt-in)**: `IndexIVFFlat` with CUDA acceleration and optimized clustering

### 📊 GPU vs CPU Comparison

| Metric | CPU (IndexFlatIP) | GPU (IndexIVFFlat) | Speedup |
|--------|-------------------|---------------------|---------|
| Search Time | ~100ms | ~5-10ms | 10-20x |
| Index Build | ~300s | ~30-60s | 5-10x |
| Memory Usage | RAM only | GPU VRAM + RAM | More efficient |
| Accuracy | 100% (exact) | 95-99% (approx) | Slight trade-off |

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

You can launch the application in either CPU or GPU mode. The compose file also mounts HF/Torch cache volumes to speed up subsequent startups.

#### **CPU Mode** (Default)
This will run on any machine with Docker installed.
```bash
docker compose -f docker/compose.yaml up --build
```
> **Note**: The first build might take a while as it downloads the base images and PyTorch binaries. Subsequent builds will be much faster. For detailed build logs, use `docker compose -f docker/compose.yaml build --progress=plain`.

#### **🎮 GPU Mode** (Optional for Performance)
This leverages your NVIDIA GPU for significantly faster inference and vector search.
```bash
docker compose -f docker/compose.yaml -f docker/compose.gpu.yaml up --build
```
This layers the GPU-specific configuration over the base setup. To allow the retriever to use GPU, ensure the backend has `FAISS_FORCE_CPU=0`.

**GPU Benefits (when enabled):**
- ⚡ **10-20x faster** vector search with CUDA-optimized FAISS
- 🏗️ **5-10x faster** index building with GPU clustering
- 🧠 **Smart memory management** with FP16 optimization
- 📈 **Better accuracy** with larger overfetch ratios
- 🔄 **Automatic fallback** to CPU if needed

### 3. Verify GPU Acceleration

Once running, check if GPU acceleration is active:

```bash
# Ensure the backend is configured to allow GPU
# (set in docker compose env or container env)
export FAISS_FORCE_CPU=0

# Check health endpoint for GPU status
curl http://localhost:8000/healthz | jq

# Expected response with GPU enabled:
{
  "status": "ok",
  "gpu_enabled": true,
  "index_type": "IndexIVFFlat",
  "num_vectors": "50000",
  "embedding_model": "intfloat/e5-base-v2",
  "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

### 4. Test GPU Performance

Run the included benchmark script to compare CPU vs GPU performance:

```bash
# Inside the GPU container
docker exec -it solorag-backend-gpu python scripts/test_gpu_faiss.py
```

This will show detailed performance comparisons and verify your GPU setup.

### 5. Customizing the LLM
By default, the application uses `tinyllama:latest` for fast startup. You can switch to any other model from the [Ollama Library](https://ollama.com/library) by setting the `OLLAMA_MODEL` environment variable.

Open `docker/compose.yaml` and modify the `environment` section for the `backend` service:
```yaml
# docker/compose.yaml
services:
  backend:
    ...
    environment:
      - OLLAMA_URL=http://localhost:11434
      # Set the model you want to use.
      - OLLAMA_MODEL=mistral:7b-instruct-q5_K_M
```
When you next run `docker compose -f docker/compose.yaml up`, the entrypoint script will automatically pull the specified model.

> ⏳ **First-time startup can take a few minutes** because Ollama may need to download LLM weights. We default to a small model to minimise this. You can monitor progress with:
>
> ```bash
> docker compose logs -f backend
> ```
>
> Once you see `SoloRAG is ready!` in the logs, both the API and UI will be available.

### 6. Access the Application
Once the container is running and you see the log `SoloRAG is ready!`, you can access the application through multiple interfaces:

**🌐 Streamlit Chat UI** (Recommended for interactive use)
Visit [http://localhost:8501](http://localhost:8501) in your browser for a chat interface.

**📚 API Documentation**
Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

**🔧 Direct API Access**
You can also send requests directly to the API:
```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "When is my first payout on Stripe?"}' | jq
```

Want chunk-by-chunk tokens? Use the streaming endpoint:
```bash
curl -N -X POST http://localhost:8000/query/stream \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I issue a refund on Stripe?"}'
```
The response will stream incrementally, followed by a `[SOURCES]` block.

---

## 🏗️ Building GPU-Optimized Indices

When building custom indices, you can force GPU optimization:

```bash
# Set BUILD_MODE to enable GPU index building
export BUILD_MODE=gpu
python scripts/build_index.py
```

This will:
- 🎯 Train IVF clusters on GPU for faster processing
- 📦 Create optimized index with better search performance  
- 🔄 Automatically fall back to CPU if GPU isn't available
- 📊 Show detailed build statistics and index information

---

## 🔧 Local Development (Without Docker)

If you prefer to run the application directly on your machine, follow these steps.

### 1. Environment Setup
Create and activate a Python virtual environment.
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

**🎯 Recommended: Use the automated installer**
```bash
# Auto-detect hardware and install appropriate packages
./scripts/install_requirements.sh

# Or force a specific mode:
./scripts/install_requirements.sh --gpu   # Force GPU setup
./scripts/install_requirements.sh --cpu   # Force CPU setup
```

**📋 Manual Installation (Advanced)**
The project uses separate requirements files for CPU and GPU to manage dependencies like PyTorch and FAISS.

```bash
# Install common packages
pip install -r requirements/common.txt

# Install hardware-specific packages
pip install -r requirements/cpu.txt     # For CPU-only setup
# OR
pip install -r requirements/gpu.txt     # For GPU setup (requires CUDA)

# Install main dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**⚠️ Important**: Don't install both `faiss-cpu` and `faiss-gpu` - they conflict with each other!

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
# For GPU acceleration (if available)
export BUILD_MODE=gpu
uvicorn app.main:app --reload
```
The API will be available at `http://localhost:8000`.

**Optional: Run Streamlit UI**
If you want to use the chat interface, start Streamlit in a separate terminal:
```bash
streamlit run streamlit_app.py
```
The UI will be available at `http://localhost:8501` and will connect to the FastAPI backend at `http://localhost:8000`.

---

## 🧪 Testing Guide

The project includes a comprehensive test suite (40+ tests) to ensure correctness and stability, covering API, retrieval, prompt logic, Ollama client, streaming, Streamlit integration, Docker setup, and **GPU acceleration**.

### Running Tests

**🎯 Recommended: Use the test runner script**
```bash
# Run all tests (GPU tests auto-skipped if hardware not available)
./scripts/run_tests.sh

# Run only GPU tests (requires GPU hardware)
./scripts/run_tests.sh --gpu

# Skip GPU tests entirely
./scripts/run_tests.sh --no-gpu

# Run with pytest options
./scripts/run_tests.sh -v -s               # Verbose output
./scripts/run_tests.sh -k "api"            # Only API tests
./scripts/run_tests.sh --gpu -v            # GPU tests with verbose output
```

**🎮 GPU Test Requirements**
GPU tests require:
- NVIDIA GPU with CUDA support
- `faiss-gpu` package installed  
- PyTorch with CUDA support
- GPU drivers and CUDA toolkit

If GPU hardware is not available, GPU tests are automatically skipped with a clear message.

### Test Suite Overview
The test suite is located in `app/tests/` and covers the following components:

*   `test_api.py`: Tests the FastAPI endpoints, including `/query` and `/query/stream` (used by Streamlit). Ensures valid/invalid requests are handled, and that streaming responses are compatible with the UI.
*   `test_retrieval.py`: Unit tests the retrieval logic. It verifies that the retriever correctly finds relevant document chunks from the FAISS index and handles cases with no matching context.
*   `test_prompt.py`: Unit tests the prompt generation logic. It checks that the final prompt sent to the LLM is correctly formatted based on the retrieved context.
*   `test_ollama_client.py`: Unit tests the asynchronous Ollama client. It mocks the `httpx` library to ensure the client correctly sends requests, handles successful responses, and manages retries upon failure.
*   `test_streaming.py`: Verifies `/query/stream` behaviour, streaming output, and logging events.
*   `test_streamlit_app.py`: Tests Streamlit integration, including environment variable handling, backend connectivity, endpoint construction, and UI compatibility with the backend API.
*   `test_docker_setup.py`: Verifies Docker Compose, Dockerfile, and entrypoint script structure, required environment variables, artifact mounting, and build/run command structure.

> **Note:** The Streamlit tests are skipped if Streamlit is not installed in your environment.

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