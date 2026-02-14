FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py train.py ./
COPY src ./src
RUN mkdir -p models

EXPOSE 7860

CMD ["python", "app.py"]
