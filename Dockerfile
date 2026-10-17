# Streamlit demo image:  docker build -t rul-demo . && docker run -p 7860:7860 rul-demo
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py .
COPY configs/ ./configs/
COPY src/ ./src/
COPY demo_assets/ ./demo_assets/

RUN useradd -m -u 1000 appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser

EXPOSE 7860
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
