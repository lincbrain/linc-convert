# ----------------------------------------------------------------------------
# STAGE 1: Builder
# Used to install Poetry, compile dependencies, and create the virtual environment.
# ----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies (GCC, etc.)
# We need these to compile packages, but we don't want them in the final image.
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.8.4

# Configure Poetry to create the venv inside the project folder
# This makes it easy to copy the whole folder to the next stage
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy dependency definition files first (for caching)
COPY pyproject.toml poetry.lock ./
RUN poetry lock
# Install dependencies (only the libraries)
RUN poetry install --no-root --no-dev --extras all && rm -rf $POETRY_CACHE_DIR

# Copy the rest of the application code
COPY . .
RUN poetry lock
# Install the application itself
#RUN poetry install --no-dev --extras all
RUN poetry build -f wheel
RUN ./.venv/bin/pip install dist/*.whl
# ----------------------------------------------------------------------------
# STAGE 2: Runtime
# A fresh, empty image that only contains the Python runtime and our copied venv.
# ----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create a non-root user for security (Best Practice)
RUN useradd -m -r appuser && chown appuser /app
USER appuser

# Copy the virtual environment from the builder stage
# We assume the path structure is identical (/app/.venv)
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Update PATH environment variable
# This allows us to type 'linc-convert' instead of '/app/.venv/bin/linc-convert'
ENV PATH="/app/.venv/bin:$PATH"
LABEL org.opencontainers.image.source=https://github.com/lincbrain/linc-convert
# Set the entrypoint
ENTRYPOINT ["linc-convert"]
# Default command
CMD ["--help"]
