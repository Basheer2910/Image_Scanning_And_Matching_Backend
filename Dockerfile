# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies and Python
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip python3.10-venv poppler-utils tesseract-ocr libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Copy only requirements.txt first (for caching)
COPY requirements.txt .

# Install Python dependencies (including Flask with async support if needed)
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Create non-root user and grant ownership of /app so user can write
RUN useradd -m appuser && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port Flask runs on
EXPOSE 5000

# Run the application with gunicorn (recommended for production) or fallback to python3 if needed
CMD ["python3", "/app/app.py"]
