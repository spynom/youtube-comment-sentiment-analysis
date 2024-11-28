FROM python:3.10
LABEL authors="spynom"

WORKDIR /app

COPY app/ /app/

RUN pip install --no-cache-dir -r requirements.txt --timeout=120 && \
    python3.10 -c "import nltk; nltk.download('stopwords')" && \
    python3.10 -c "import nltk; nltk.download('wordnet')"

RUN python3.10 model_update.py


EXPOSE 8000

CMD ["python3.10","app.py"]