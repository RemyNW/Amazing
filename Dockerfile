FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# COPY data/bronze ./data/bronze
COPY scripts/ ./scripts/

CMD ["python", "scripts/convert_csv_parquet.py"]
