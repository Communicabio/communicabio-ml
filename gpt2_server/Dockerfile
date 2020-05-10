# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ru-GPT2Like /app/ru-GPT2Like

COPY *.py /app/

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD cd /app && exec gunicorn --bind :$PORT --workers 2 --threads 2 --timeout 0 app:app
