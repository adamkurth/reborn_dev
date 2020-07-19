FROM ubuntu:20.04
RUN apt-get -qq -y update
# RUN apt-get -qq -y install apt-utils
RUN apt-get -qq -y install curl libgl1-mesa-glx ocl-icd-opencl-dev python3-pyopencl
RUN python3 -c 'import pyopencl; print(pyopencl.get_platforms())'
RUN mkdir /tmp/reborn
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --output /tmp/reborn/miniconda.sh
RUN bash /tmp/reborn/miniconda.sh -b -p /usr/local/miniconda
RUN /usr/local/miniconda/bin/conda update -n base -c defaults conda
COPY environment.yml /tmp/reborn/environment.yml
RUN /usr/local/miniconda/bin/conda env create --name reborn --file /tmp/reborn/environment.yml
RUN mv /usr/local/miniconda/envs/reborn/etc/OpenCL/vendors/pocl.icd /usr/local/miniconda/envs/reborn/etc/OpenCL/vendors/pocl-conda.icd  # The opencl icd from conda is broken!!!
RUN ln -s /etc/OpenCL/vendors/pocl.icd /usr/local/miniconda/envs/reborn/etc/OpenCL/vendors/pocl.icd  # Use the working opencl icd from ubuntu apt
RUN /usr/local/miniconda/envs/reborn/bin/python -c 'import pyopencl; print(pyopencl.get_platforms())'
RUN /usr/local/miniconda/envs/reborn/bin/python -c 'import OpenGL'
RUN /usr/local/miniconda/envs/reborn/bin/python -c 'import PyQt5'