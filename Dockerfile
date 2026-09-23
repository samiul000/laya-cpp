FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build tools, compilers, and python for verification
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies for parity testing
RUN pip3 install --no-cache-dir \
    onnxruntime==1.23.2 \
    numpy

# Download pre-built ONNX Runtime C++ library (Linux x64)
WORKDIR /opt
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz && \
    tar -xzf onnxruntime-linux-x64-1.23.2.tgz && \
    rm onnxruntime-linux-x64-1.23.2.tgz

ENV ORT_HOME=/opt/onnxruntime-linux-x64-1.23.2
ENV LD_LIBRARY_PATH=${ORT_HOME}/lib

WORKDIR /workspace
CMD ["/bin/bash"]