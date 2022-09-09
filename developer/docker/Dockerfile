FROM ubuntu:20.04
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -q -y update && \
    apt-get -q -y install apt-utils curl make gfortran pocl-opencl-icd libgl1-mesa-glx && \
    apt-get -q -y autoremove && \
    apt-get -q -y clean
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -q -y install texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-science latex2html
RUN mkdir /tmp/reborn
COPY environment.yml /tmp/reborn/environment.yml
RUN export DEBIAN_FRONTEND=noninteractive && \
    curl -s https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --output /tmp/reborn/miniconda.sh && \
    bash /tmp/reborn/miniconda.sh -b -p /usr/local/miniconda && \
    /usr/local/miniconda/bin/conda update -n base -c defaults conda && \
    /usr/local/miniconda/bin/conda env create --name reborn --file /tmp/reborn/environment.yml nomkl && \
    /usr/local/miniconda/bin/conda create -y --name reborn-minimal nomkl python=3.7 scipy && \
    /usr/local/miniconda/bin/conda clean -y --all && \
    rm -r /tmp/reborn /usr/local/miniconda/pkgs
# We use the pocl.icd from apt because the pocl.icd from from conda is broken as of July 2020!
RUN ln -s /etc/OpenCL/vendors/pocl.icd /usr/local/miniconda/envs/reborn/etc/OpenCL/vendors/pocl.icd
RUN /usr/local/miniconda/envs/reborn/bin/python -c 'import PyQt5, OpenGL, pyopencl; print(pyopencl.get_platforms())'
# The following is for the sphinx-gallery qt scraper.  Hopefully we can remove the dependency.
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -q -y install xvfb