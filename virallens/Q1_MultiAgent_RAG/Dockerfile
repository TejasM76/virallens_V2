# Multi-stage build — pre-downloads embedding model for offline use
FROM python:3.12-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY q1_multiagent_rag/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

FROM python:3.12-slim AS runtime
WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy project layout
COPY q1_multiagent_rag/ ./q1_multiagent_rag/
COPY q2_custom_embedder/ ./q2_custom_embedder/
COPY run.py .
COPY .env .env

# data/ is mounted at runtime — don't bake PDFs into the image
RUN mkdir -p data .vectorstore

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=10s CMD python -c "import chromadb, langchain, groq; print('OK')" || exit 1

# Default: run Q1 demo queries
ENTRYPOINT ["python", "run.py"]
CMD ["q1"]
