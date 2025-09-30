# Stage 1: Build dependencies
FROM python:3.12-slim as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ curl

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# Stage 2: Final image
FROM python:3.12-slim

WORKDIR /app

# Install curl for healthcheck
# RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN addgroup --system app && adduser --system --group --home /app app

# Copy built dependencies
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy application code
COPY ./app ./app

# Copy Aerich configuration and migrations
COPY ./pyproject.toml ./pyproject.toml
COPY ./migrations ./migrations

# Set ownership
RUN mkdir /app/.dspy_cache
RUN chown -R app:app /app

# Switch to non-root user
USER app

ENV DSPY_CACHE_DIR /app/.dspy_cache

# Expose port and run the application
EXPOSE 8080
ENV WORKERS 4
CMD gunicorn -k uvicorn.workers.UvicornWorker -w $WORKERS -b 0.0.0.0:8080 app.main:app

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
