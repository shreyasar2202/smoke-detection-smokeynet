ARG cuda_version=10.1
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

WORKDIR /userdata/kerasData

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 \
    g++ \
    git \
    graphviz \
    gifsicle \
    ninja-build \
    libgl1-mesa-glx \
    libhdf5-dev \
    sudo\
    strace \
    openmpi-bin \
    protobuf-compiler \
    xvfb \
    screen \
    vim \
    emacs \
    nano \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Install Python packages and keras
ENV NB_USER keras
ENV NB_UID 1000
# RUN mkdir /userdata/kerasData
RUN echo "root:digits" | chpasswd

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    chown $NB_USER $CONDA_DIR -R && \
    chown $NB_USER /userdata/kerasData -R && \
    chown $NB_USER / && \
    mkdir -p / && \
    chpasswd $NB_USER:digits && \
    usermod -aG sudo $NB_USER

USER $NB_USER

ARG python_version=3.6

RUN conda config --append channels conda-forge
RUN conda install -y python=${python_version} && \
    pip install --upgrade pip && \
    pip install \
    sklearn_pandas \
    opencv-python \
    pycocotools>=2.0.1 \
    google-colab \
    scikit-image \
    conda install \
    h5py \
    matplotlib \
    jupyterlab \
    Pillow \
    pandas \
    pyyaml \
    scikit-learn \
    && \
    conda clean -yt

RUN pip install --upgrade ipykernel

# RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
# Or using pip
RUN pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install detectron2==0.2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHONPATH='/src/:$PYTHONPATH'

EXPOSE 8888

#RUN mkdir /home/$NB_USER/.jupyter
#ADD docker/jupyter_notebook_config.py /home/$NB_USER/.jupyter

# Run time ENV for port of jupyter lab, if not defined, default to 8888
#ARG MY_JUPYTER_LAB_PORT=8888
#ENV MY_JUPYTER_LAB_PORT="${MY_JUPYTER_LAB_PORT}"

#CMD jupyter lab --port=${MY_JUPYTER_LAB_PORT} --no-browser --ip=0.0.0.0 --allow-root
