FROM python:3.7-slim

RUN pip install torch==1.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ru-GPT2Like /app/ru-GPT2Like/
COPY vocab.bpe /app/

COPY *.py /app/
RUN ls /app

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD cd /app && gunicorn --bind :8080 --workers 1 --threads 4 --timeout 0 app:app
