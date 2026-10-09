FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# models/ must exist: run `python -m app.main` before building the image.
COPY app ./app
COPY data/raw ./data/raw
COPY models ./models

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
