# syntax=docker/dockerfile:1

# --- Build stage: compile a wheel with the api + formats extras -------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Copy only what the build backend (hatchling) needs to resolve the version and
# assemble the wheel.
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src

# Build wheels for Sorbed and all its runtime dependencies into /wheels so the
# final image installs them without a compiler toolchain present.
RUN pip install --upgrade pip build \
    && pip wheel --wheel-dir /wheels ".[api,formats]"

# --- Runtime stage: lean image, non-root -----------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SORBED_OUTPUT_DIR=/data/reports

# opencv-python-headless still needs libglib2.0 at runtime; curl backs the
# healthcheck.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels "sorbed[api,formats]" \
    && rm -rf /wheels

# Run as an unprivileged user and give it a writable output directory.
RUN useradd --create-home --uid 10001 sorbed \
    && mkdir -p /data/reports /models \
    && chown -R sorbed:sorbed /data /models
USER sorbed
WORKDIR /home/sorbed

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/v1/health || exit 1

CMD ["uvicorn", "sorbed.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
