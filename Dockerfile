FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .
COPY multisync/ multisync/
COPY syncpipe/ syncpipe/
COPY tests/ tests/

RUN pip install --no-cache-dir -e ".[dev]"

RUN python -m pytest tests/ -q --tb=short

CMD ["python", "-m", "syncpipe", "--version"]
