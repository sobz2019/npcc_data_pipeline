FROM apache/airflow:2.10.2

USER root

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Switch to airflow user for pip installations
USER airflow

# Install Python dependencies with more error handling
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir apache-airflow-providers-apache-spark==2.1.3 apache-airflow-providers-amazon==7.3.0