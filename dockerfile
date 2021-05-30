FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu16.04

SHELL ["/bin/bash", "-c"] 

##### PYENV & PYTHON #####
# Install pyenv dependencies & fetch pyenv
# see: https://github.com/pyenv/pyenv/wiki/common-build-problems
RUN apt-get update && \
    apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git && \
    apt-get install -y libsndfile1 && \
    git clone --single-branch --depth 1  https://github.com/pyenv/pyenv.git /.pyenv

RUN apt-add-repository ppa:mc3man/trusty-media
RUN apt-get update
RUN apt-get install -y ffmpeg

ENV PYENV_ROOT="/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"
ENV PATH="$PYENV_ROOT/shims:$PATH"

ARG PYTHON_VERSION=3.8.9

RUN pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION}


##### PYTHON PACKAGE DEPENDENCIES #####
WORKDIR /app
COPY ./pyproject.toml /app/
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install
RUN pip install poethepoet
RUN poe force-cuda11 
RUN pip install fastapi uvicorn

##### APPLICATION #####
COPY . /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
