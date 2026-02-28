FROM python:3.11-slim

# Install system dependencies required for pgvector and psycopg2
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install fastapi uvicorn pydantic numpy anthropic sqlalchemy psycopg2-binary pgvector

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
