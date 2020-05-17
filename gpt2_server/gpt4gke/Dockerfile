# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM gcr.io/deeplearning-platform-release/pytorch-gpu

COPY requirements.txt .
RUN pip install -r requirements.txt

#RUN  apt-get update && apt-get install -y wget
#RUN apt-get update && apt-get install -y gnupg2
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb && \
#    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
#    dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb && \
#    apt-get update && \
#    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb && \
#    apt install -y ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb && \
#    apt-get update
#RUN echo "31\n" | apt-get install -y --no-install-recommends nvidia-driver-430

COPY ru-GPT2Like /app/ru-GPT2Like
COPY vocab.bpe /app/
COPY *.py /app/
RUN ls /app

CMD cd /app && python3 app.py
