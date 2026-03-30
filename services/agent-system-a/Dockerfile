# Build from repository root:
#   docker compose build agent-system-a
# Preserves services/agent-system-a path so PROJECT_ROOT (/itip) resolves correctly.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /itip

COPY services/agent-system-a/requirements.txt /itip/services/agent-system-a/requirements.txt

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /itip/services/agent-system-a/requirements.txt --index-url https://pypi.org/simple

COPY services/agent-system-a /itip/services/agent-system-a

WORKDIR /itip/services/agent-system-a

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
