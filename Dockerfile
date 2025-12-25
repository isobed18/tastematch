# Use Python 3.11 Slim (Debian-based)
FROM python:3.11-slim

# Install system build dependencies (required for compiling LightFM extensions if wheel missing)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Install Python dependencies
# - lightfm: Recommender model
# - pandas: Data manipulation
# - scipy: Sparse matrices
# - numpy: Numerical ops
# - matplotlib: Plotting
# - scikit-learn: Metrics/Splitting
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    lightfm

# Default command matches typical training args (can be overridden)
CMD ["python", "project/fm/train.py", "--help"]
