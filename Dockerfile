FROM python:3.7
WORKDIR /src

COPY . /src
RUN pip install --no-cache-dir --requirement /src/requirements.txt
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader punkt

CMD ["python3", "main.py", "8080"]
