FROM python:3.13.5-slim

# Install uv and curl (needed for HEALTHCHECK)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install uv==0.5.21

# Create a non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install exact dependencies securely (no dev dependencies, do not install local project to avoid needing src/ yet)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY src/ ./src/
COPY registry/ ./registry/

# Change ownership of /app to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Environment variables for API config (can be overridden at runtime)
ENV HOST=0.0.0.0
ENV PORT=8000
ENV CORS_ORIGINS=""
# Ensure we use the virtual environment created by uv
ENV PATH="/app/.venv/bin:$PATH"

# Expose the configurable port
EXPOSE $PORT

# Healthcheck to ensure the container liveness probe hits the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1

# Command to run the application using uvicorn
CMD ["sh", "-c", "uvicorn src.api.main:app --host $HOST --port $PORT"]
