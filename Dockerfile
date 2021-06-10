FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

WORKDIR /userdata/kerasData

# Install system packages
RUN apt-get update && apt-get upgrade
RUN apt-get install -y --no-install-recommends \
    python3.8 \
    git \
    sudo\
    tmux \
    wget \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install pip packages
RUN pip install --upgrade pip
RUN pip install \
    jupyterlab \
    sklearn \
    matplotlib

ENV PYTHONPATH='/src/:$PYTHONPATH'

# Set up ipynb server
EXPOSE 8888

ARG MY_JUPYTER_LAB_PORT=8888
ENV MY_JUPYTER_LAB_PORT="${MY_JUPYTER_LAB_PORT}"

# WARNING: Dockerfile will not run without this line!
CMD jupyter lab --port=${MY_JUPYTER_LAB_PORT} --no-browser --ip=0.0.0.0 --allow-root
