# ---- Stage 1: builder (packages install/build) ----
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements-prod.txt .
RUN pip install --user --no-cache-dir -r requirements-prod.txt


# ---- Stage 2: runtime (finished packages + app)
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH


COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]