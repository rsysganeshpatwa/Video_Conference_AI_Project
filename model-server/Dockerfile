# ✅ Use NVIDIA CUDA + cuDNN base image with Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ✅ Set ENV variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    GGML_CUDA=1 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
    

# ✅ Set working directory
WORKDIR /app

# ✅ Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    curl \
    wget \
    libsndfile1 \
    build-essential \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    ca-certificates && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    rm -rf /var/lib/apt/lists/*

# ✅ Copy and install requirements early to leverage Docker cache
COPY requirements.txt .

# ✅ Upgrade pip and install Python deps
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ✅ Install CTranslate2 (GPU) and faster-whisper explicitly
RUN pip install --no-cache-dir "ctranslate2[gpu]==4.4.0" faster-whisper

# ✅ Add CUDA stub symlinks so llama-cpp-python can detect GPU
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so && \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf && \
    ldconfig

# ✅ Install llama-cpp-python with CUDA support
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir --force-reinstall llama-cpp-python

# ✅ Copy the app code after deps to avoid rebuilding on every change
COPY . .

# ✅ Default command
CMD ["python", "main.py"]

