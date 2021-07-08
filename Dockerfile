# WARNING: CUDA v11 may lead to memory issues
ARG cuda_version=10.2
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

WORKDIR /userdata/kerasData

# Install system packages. Put on separate lines to use caching.
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y python3-dev 
RUN apt-get install -y python3-pip
RUN apt-get install -y git
RUN apt-get install -y sudo
RUN apt-get install -y tmux 
RUN apt-get install -y wget
RUN apt-get install -y curl
# opencv-python requirements
RUN apt-get install -y ffmpeg
RUN apt-get install -y libsm6
RUN apt-get install -y libxext6
RUN rm -rf /var/lib/apt/lists/*

# Install pip3 packages. Put on separate lines to use caching.
RUN pip3 install --upgrade pip
RUN pip3 install jupyterlab 
RUN pip3 install numpy
RUN pip3 install matplotlib
RUN pip3 install torch
RUN pip3 install torchvision
RUN pip3 install pytorch-lightning
RUN pip3 install torchmetrics
RUN pip3 install sklearn
RUN pip3 install opencv-python
RUN pip3 install transformers
RUN pip3 install einops

ENV PYTHONPATH='/src/:$PYTHONPATH'

# Set up Jupyter Lab server
EXPOSE 8888

ARG MY_JUPYTER_LAB_PORT=8888
ENV MY_JUPYTER_LAB_PORT="${MY_JUPYTER_LAB_PORT}"

# WARNING: Dockerfile will give CrashLoopBackOff error without this line!
# Password: digits
CMD jupyter lab --port=${MY_JUPYTER_LAB_PORT} --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.password="$(echo digits | python3 -c 'from notebook.auth import passwd;print(passwd(input()))')"  --ContentsManager.allow_hidden=True
