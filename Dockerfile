FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AGENT_TEAM_DB_PATH=/data/team.db \
    AGENT_TEAM_TEAM_NAME=agent-team

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples

RUN pip install --upgrade pip \
    && pip install -e .

RUN mkdir -p /data

VOLUME ["/data"]

# Start with interactive mode by default
CMD ["python", "-m", "agent_team"]
