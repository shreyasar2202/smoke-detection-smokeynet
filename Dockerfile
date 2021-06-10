FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

WORKDIR /userdata/kerasData

# Install system packages
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y \
    python3-dev \
    python3-pip \
    git \
    sudo\
    tmux \
    wget \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install pip3 packages
RUN pip3 install --upgrade pip
RUN pip3 install jupyterlab

ENV PYTHONPATH='/src/:$PYTHONPATH'

# Set up Jupyter Lab server
EXPOSE 8888

ARG MY_JUPYTER_LAB_PORT=8888
ENV MY_JUPYTER_LAB_PORT="${MY_JUPYTER_LAB_PORT}"

# WARNING: Dockerfile will not run without this line!
CMD jupyter lab --port=${MY_JUPYTER_LAB_PORT} --no-browser --ip=0.0.0.0 --allow-root
