FROM python:3.11.9-slim

WORKDIR /app/backend
COPY . .


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        build-essential \
        gcc \
        g++


RUN pip install --upgrade pip && \
    pip install --ignore-installed --no-cache-dir -r requirements.txt


EXPOSE 8500
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8500" , "--reload"]
