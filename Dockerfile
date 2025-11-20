# Dockerfile for AI Dataset Verifier – Shell-Wrapped CMD (Permission Fix)
FROM python:3.12-slim  # Stable 3.12 base

# Install minimal system deps
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /code

# Copy requirements
COPY requirements.txt .

# Install Torch/Torchvision with CPU index (wheels only)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --only-binary=all \
        torch==2.8.0+cpu torchvision==0.23.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# Install rest of deps with wheels only
RUN pip install --no-cache-dir --only-binary=all -r requirements.txt

# Copy app code
COPY . .

# Expose Streamlit port
EXPOSE 7860

# Run Streamlit via shell-wrapped module (bypasses permission)
CMD ["sh", "-c", "python -m streamlit run app.py --server.port 7860 --server.address 0.0.0.0"]
