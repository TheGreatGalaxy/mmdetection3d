ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip3 config set global.index-url http://mirrors.aliyun.com/pypi/simple
RUN pip3 config set install.trusted-host mirrors.aliyun.com
# Install MMCV, MMDetection and MMSegmentation
RUN pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install mmdet==2.17.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install mmsegmentation==0.18.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install MMDetection3D
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection3d.git /mmdetection3d
WORKDIR /mmdetection3d
ENV FORCE_CUDA="1"
# RUN pip install -r requirements/build.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install --no-cache-dir -e .
