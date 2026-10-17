# Demo image: Streamlit app for RUL prediction (used by the Hugging Face Docker Space)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

COPY streamlit_app.py .
COPY demo_assets/ ./demo_assets/

# HF Spaces runs the container as uid 1000
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
