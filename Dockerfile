# Use official PyTorch image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Expose ports for FastAPI (8000) and Gradio (7860)
EXPOSE 8000
EXPOSE 7860

# Default command: Start the FastAPI service and Gradio app using a process manager or simple script
CMD ["sh", "-c", "python service.py & python app.py"]
