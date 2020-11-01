FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04


RUN apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        python3 \
        gnupg2 \
        tar \
        ssh \
        python3-dev && \
        rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p miniconda
ENV PATH="/miniconda/bin:${PATH}"
#ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN conda config --add channels pytorch
RUN conda install scikit-learn
RUN conda install -c serge-sans-paille gcc_49
RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

RUN CC=gcc-4.9 pip --no-cache-dir install pytest nmslib seldon-core flask matplotlib numpy

ENV PYTHONPATH /miniconda/lib:/miniconda/lib/python3.7/site-packages/:/app


COPY ./ /app
WORKDIR /app
EXPOSE 5000

# Define environment variable
ENV CONFIG_PATH /app/config/app.json
CMD python /app/simtool/api.py
