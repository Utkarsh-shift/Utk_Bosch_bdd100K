# ==============================================================
# Base image: lightweight Python (no CUDA yet)
# ==============================================================
FROM python:3.10-slim

# Avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# ==============================================================
# Install essential Linux packages
# ==============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget unzip ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================
# Set working directory
# ==============================================================
WORKDIR /app

# ==============================================================
# Copy project files
# ==============================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==============================================================
# Create default folders
# ==============================================================
RUN mkdir -p /app/data /app/checkpoints /app/outputs

# ==============================================================
# Set environment
# ==============================================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ==============================================================
# Copy rest of app
# ==============================================================
COPY . /app

# ==============================================================
# Default command
# ==============================================================
CMD ["bash"]

