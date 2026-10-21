FROM python:3.12-slim

WORKDIR /srv

# Copy dependency definition first so Docker can cache the install layer.
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
