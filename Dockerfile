FROM python:3.11-slim

COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "server.py"]

CMD ["python", "server.py"]
