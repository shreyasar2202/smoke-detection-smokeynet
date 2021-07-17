# WARNING: CUDA v11 may lead to memory issues
ARG cuda_version=10.2
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

WORKDIR /userdata/kerasData

# Prevents asking for user input when installing packages
ARG DEBIAN_FRONTEND=noninteractive

# Install system packages. Put on separate lines to use caching.
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y git
RUN apt-get install -y sudo
RUN apt-get install -y tmux 
RUN apt-get install -y wget
RUN apt-get install -y curl

# opencv-python requirements
RUN apt-get install -y ffmpeg
RUN apt-get install -y libsm6
RUN apt-get install -y libxext6

# Install python3.9
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install -y python3.9

# Install pip3.9
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN apt-get install -y python3.9-distutils
RUN python3.9 get-pip.py
RUN rm -rf /var/lib/apt/lists/*

# Alias python and pip
RUN echo "alias python=python3.9" >> ~/.bashrc
RUN echo "alias pip=pip3.9" >> ~/.bashrc
RUN source ~/.bashrc

# Install pip packages. Put on separate lines to use caching.
RUN pip install --upgrade pip
RUN pip install jupyterlab 
RUN pip install numpy
RUN pip install matplotlib
RUN pip install torch
RUN pip install torchvision
RUN pip install pytorch-lightning
RUN pip install torchmetrics
RUN pip install sklearn
RUN pip install opencv-python
RUN pip install transformers
RUN pip install efficientnet-pytorch

# Fix ipynb bug with progress bar
RUN pip install ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension

ENV PYTHONPATH='/src/:$PYTHONPATH'

# Set up Jupyter Lab server
EXPOSE 8888

ARG MY_JUPYTER_LAB_PORT=8888
ENV MY_JUPYTER_LAB_PORT="${MY_JUPYTER_LAB_PORT}"

# WARNING: Dockerfile will give CrashLoopBackOff error without this line!
# Password: digits
CMD jupyter lab --port=${MY_JUPYTER_LAB_PORT} --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.password="$(echo digits | python -c 'from notebook.auth import passwd;print(passwd(input()))')"  --ContentsManager.allow_hidden=True
