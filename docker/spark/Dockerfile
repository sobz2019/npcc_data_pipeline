FROM bitnami/spark:latest

USER root

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies with more error handling
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Set back to non-root user
USER 1001