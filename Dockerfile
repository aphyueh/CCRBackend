# backend/Dockerfile
FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .

CMD ["gunicorn", "-b", ":8080", "main:app"]
