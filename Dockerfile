FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev deps, no cache)
RUN uv sync --frozen --no-cache

# Copy source code
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed data/chroma

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Default: run Streamlit (override in ECS task definition for API)
CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
