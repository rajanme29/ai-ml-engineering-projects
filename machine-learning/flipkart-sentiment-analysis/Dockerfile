#base image
FROM python:3.11-slim-bookworm

#env variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# cwd
WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

#copy all necessary project files
COPY --chown=appuser:appgroup api/ ./api/
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup model/ ./model/

#expose FastAPI port
EXPOSE 8000

#Start API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]