FROM python:3.11-slim

WORKDIR /src

RUN apt-get update && apt-get install -y git

COPY . .

CMD ["unicorn", "app:index", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"]
