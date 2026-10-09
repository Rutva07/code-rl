FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml /app/
COPY src /app/src
RUN pip install --no-cache-dir . && useradd --create-home --uid 10001 coderl && mkdir /worker && chown coderl:coderl /worker
USER coderl
ENV CODERL_WORKER_DIR=/worker
CMD ["coderl", "serve", "--host", "0.0.0.0", "--port", "8000"]
