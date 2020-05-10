FROM python:3.7-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY rubert-base-uncased /app/rubert-base-uncased

COPY *.py /app/
RUN ls /app

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD cd /app && gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
