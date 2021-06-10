FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

WORKDIR /userdata/kerasData

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    sudo\
    nano \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Configure user
ENV NB_USER keras
ENV NB_UID 1000
RUN echo "root:digits" | chpasswd

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    chown $NB_USER /userdata/kerasData -R && \
    chown $NB_USER / && \
    mkdir -p / && \
    chpasswd $NB_USER:digits && \
    usermod -aG sudo $NB_USER

USER $NB_USER

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHONPATH='/src/:$PYTHONPATH'

EXPOSE 8888
