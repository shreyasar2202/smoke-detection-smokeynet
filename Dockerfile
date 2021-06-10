FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

WORKDIR /userdata/kerasData

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    sudo\
    tmux \
    nano \
    wget && \
    rm -rf /var/lib/apt/lists/*
