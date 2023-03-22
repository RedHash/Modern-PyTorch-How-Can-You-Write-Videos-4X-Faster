FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV ccap=75

RUN apt-get update -y && \
    apt-get install -y python3-pip libgl1 libglib2.0-0 libsndfile1 build-essential yasm cmake libtool libc6 libc6-dev unzip libnuma1 libnuma-dev git zip curl cron wget nginx pkg-config libx264-dev libgnutls28-dev tzdata

RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && cd nv-codec-headers && git checkout n11.0.10.1 && make install
RUN wget -q https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n4.4.2.tar.gz && tar -xf n4.4.2.tar.gz && mv FFmpeg-n4.4.2 ffmpeg && cd ffmpeg && ./configure \
  --prefix='/usr/' \
  --extra-cflags='-I/usr/local/cuda/include' \
  --extra-ldflags='-L/usr/local/cuda/lib64' \
  --nvccflags="-gencode arch=compute_${ccap},code=sm_${ccap} -O2" \
  --enable-decoder=aac \
  --enable-decoder=h264 \
  --enable-decoder=h264_cuvid \
  --enable-decoder=rawvideo \
  --enable-indev=lavfi \
  --enable-encoder=libx264 \
  --enable-encoder=h264_nvenc \
  --enable-demuxer=mov \
  --enable-muxer=mp4 \
  --enable-filter=scale \
  --enable-filter=testsrc2 \
  --enable-protocol=file \
  --enable-protocol=https \
  --enable-gnutls \
  --enable-shared \
  --enable-gpl \
  --enable-nonfree \
  --enable-cuda-nvcc \
  --enable-libx264 \
  --enable-nvenc \
  --enable-cuvid \
  --enable-nvdec \
  --enable-gpl
RUN cd ffmpeg && make clean && make -j > /dev/null 2>&1 && make install

ADD requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /root/

CMD /bin/bash run.sh