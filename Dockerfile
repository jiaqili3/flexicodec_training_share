# docker build -t pytorch-flash-attn .

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ARG VENV_NAME="tts"
ENV VENV=$VENV_NAME
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEN=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "--login", "-c"]

# RUN apt-get update -y --fix-missing
# RUN apt-get install -y git build-essential curl wget ffmpeg unzip git sox libsox-dev libsndfile1 && \
#     apt-get clean

# ==================================================================
# conda install and conda forge channel as default
# ------------------------------------------------------------------
# Install miniforge
# RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
#     /bin/bash ~/miniforge.sh -b -p /opt/conda && \
#     rm ~/miniforge.sh && \
#     ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#     echo "source /opt/conda/etc/profile.d/conda.sh" >> /opt/nvidia/entrypoint.d/100.conda.sh && \
#     echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#     echo "conda activate ${VENV}" >> /opt/nvidia/entrypoint.d/110.conda_default_env.sh && \
#     echo "conda activate ${VENV}" >> $HOME/.bashrc

# ENV PATH /opt/conda/bin:$PATH

# RUN conda config --add channels conda-forge && \
#     conda config --set channel_priority strict
# # ------------------------------------------------------------------
# # ~conda
# # ==================================================================

# RUN conda create -y -n ${VENV} python=3.10
# ENV CONDA_DEFAULT_ENV=${VENV}
# ENV PATH /opt/conda/bin:/opt/conda/envs/${VENV}/bin:$PATH

WORKDIR /ssd2/lijiaqi18
RUN pip install ninja
RUN python -m pip install --upgrade pip wheel setuptools
RUN MAX_JOBS=64 python -m pip -v install flash-attn --no-build-isolation

ENV PYTHONPATH="${PYTHONPATH}"

# Install Python dependencies (requirements.txt) if needed
COPY /ssd2/
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt


# RUN conda activate ${VENV} && conda install -y -c conda-forge pynini==2.1.5
WORKDIR /workspace