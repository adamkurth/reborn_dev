FROM ubuntu:20.04
RUN apt-get -qq -y update
RUN apt-get -qq -y install apt-utils curl libgl1-mesa-glx
RUN mkdir /tmp/reborn
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --output /tmp/reborn/miniconda.sh
RUN bash /tmp/reborn/miniconda.sh -b -p /usr/local/miniconda
RUN /usr/local/miniconda/bin/conda update -n base -c defaults conda
COPY environment.yml /tmp/reborn/environment.yml
RUN /usr/local/miniconda/bin/conda env update --name base --file /tmp/reborn/environment.yml
RUN echo export PATH=/usr/local/miniconda/bin:$PATH >> ~/.bashrc
RUN /usr/local/miniconda/bin/python -c 'import pyopencl, OpenGL, PyQt5'
