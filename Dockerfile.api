#Steps involved:
#1. Start with a base OS (FROM)
#2. Install system tools (RUN apt-get)
#3. Set a working directory (WORKDIR)
#4. Copy dependency list (COPY requirements.txt)
#5. Install Python packages (RUN pip install)
#6. Copy your actual code (COPY)
#7. Say which port to listen on (EXPOSE)
#8. Say what command to run (CMD)

#Stage 1 - Build image
FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


#Stage 2 - Runtime image
FROM python:3.11-slim AS runtime
WORKDIR /app

COPY --from=builder /install /usr/local
COPY app.py .
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY models/ ./models/

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
 
