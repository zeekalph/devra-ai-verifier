# Dockerfile for AI Dataset Verifier – Force Pandas Wheel + Torch CPU
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
        torch==2.8.0+cpu torchvision==0.23.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir uvicorn --target /usr/local/bin
# Install streamlit globally (PATH + permissions fix)
RUN pip install --no-cache-dir streamlit --target /usr/local/bin && \
    chmod +x /usr/local/bin/streamlit

    
# Copy app code
COPY . .

# Expose Streamlit port
EXPOSE 7860

# Run Streamlit (UI + FastAPI in background)
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
