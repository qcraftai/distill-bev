
FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda install python=3.8
RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
RUN conda install pytorch-scatter -c pyg
RUN pip install Pillow==8.4.0
RUN pip install tqdm
RUN pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12/index.html
RUN pip install mmdet==2.24.0
RUN pip install mpi4py==3.0.3
RUN pip install numba==0.48.0
RUN pip install opencv-python
RUN pip install timm==0.5.4
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
RUN pip install hydra-core
RUN pip install spconv-cu116
RUN pip install terminaltables
RUN pip install nuscenes-devkit
RUN pip install lyft-dataset-sdk
RUN pip install einops
RUN pip install trimesh
RUN pip install pypcd
RUN pip install ipython
RUN pip install future tensorboard