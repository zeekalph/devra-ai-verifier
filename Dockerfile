# Dockerfile for AI Dataset Verifier – Force Wheel Installs (No Source Build)
FROM python:3.12.11

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
        torch==2.2.0+cpu torchvision==0.17.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# Install rest of deps with wheels only (no source build)
RUN pip install --no-cache-dir --only-binary=all -r requirements.txt

# Copy app code
COPY . .

# Expose Streamlit port
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
