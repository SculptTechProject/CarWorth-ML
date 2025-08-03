FROM python:3.11-slim
LABEL maintainer="Mateusz Dalke"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "src.main", "--data", "data/cars_5m.csv.gz", "--out", "models/model.joblib", "--cv", "--plots"]