FROM python:3.9-slim

WORKDIR /app
COPY docker/requirements.txt /tmp/requirements.txt
COPY streamlit/app.py .

RUN pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
