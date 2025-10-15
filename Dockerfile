# ===== builder: install deps =====
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel -r requirements.txt -w wheels

# ===== runtime =====
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app
COPY --from=builder /app/wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/* && rm -rf /wheels

# code & metadata
COPY VERSION .
COPY src ./src
COPY train.py .
# 模型在构建阶段训练或在CI中产出后COPY
COPY models ./models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "diabetes_service.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
