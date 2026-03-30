# Build from repository root:
#   docker compose build agent-system-b

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /itip

COPY services/agent-system-b/requirements.txt /itip/services/agent-system-b/requirements.txt
RUN pip install --no-cache-dir -r /itip/services/agent-system-b/requirements.txt

COPY services/agent-system-b /itip/services/agent-system-b

WORKDIR /itip/services/agent-system-b

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
