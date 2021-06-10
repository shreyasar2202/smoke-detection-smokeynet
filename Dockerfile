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


ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHONPATH='/src/:$PYTHONPATH'


EXPOSE 8888
