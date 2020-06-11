FROM python:3.7-slim

RUN pip install torch==1.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY gpt2 /app/gpt2

COPY *.py /app/
RUN ls /app

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD cd /app && python3 app.py
