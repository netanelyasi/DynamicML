FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

RUN mkdir -p uploads models && \
    chmod +x start.sh

ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app

CMD ["bash", "start.sh"]
