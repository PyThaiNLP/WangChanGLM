FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04

RUN ["/bin/bash", "-c", "echo I am using bash"]
SHELL ["/bin/bash", "-c"]
##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}

##############################################################################
# Installation/Basic Utilities
##############################################################################
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common build-essential autotools-dev \
    nfs-common pdsh \
    cmake g++ gcc \
    curl wget vim tmux emacs less unzip \
    htop iftop iotop ca-certificates openssh-client openssh-server \
    rsync iputils-ping net-tools sudo \
    llvm-11-dev

##############################################################################
# Installation Latest Git
##############################################################################
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get update && \
    apt-get install -y git && \
    git --version
##############################################################################
# Python
##############################################################################
ENV PYTHON_VERSION=3
RUN apt-get install -y python3 python3-dev && \
    rm -f /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    pip install --upgrade pip && \
    # Print python an pip version
    python -V && pip -V
RUN pip install pyyaml
RUN pip install ipython

##############################################################################
# PyTorch
##############################################################################
ENV PYTORCH_VERSION=1.13.1
ENV TORCHVISION_VERSION=0.14.1
ENV TENSORBOARDX_VERSION=2.6
RUN pip install torch==${PYTORCH_VERSION}
RUN pip install torchvision==${TORCHVISION_VERSION}
RUN pip install tensorboardX==${TENSORBOARDX_VERSION}

##############################################################################
# DeepSpeed
##############################################################################
RUN git clone https://github.com/qywu/apex
WORKDIR ./apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
RUN pip install triton
RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip install deepspeed
RUN pip install bitsandbytes
RUN python -c "import deepspeed; print(deepspeed.__version__)" && ds_report

WORKDIR /workspace
RUN echo I am using bash, which is now the default
ENV SHELL=/bin/bash
RUN pip install jupyter -U && pip install jupyterlab
EXPOSE 8888
ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
