FROM ubuntu:18.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        g++ \
        make \
        wget \
        unzip \
        vim \
        git \
        libssl-dev

WORKDIR /tmp
# Install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.6/cmake-3.18.6.tar.gz \
    && tar xvf cmake-3.18.6.tar.gz && cd cmake-3.18.6 \
    && ./bootstrap && make -j$(nproc) && make install

# Install SEAL
RUN wget https://github.com/microsoft/SEAL/archive/refs/tags/v3.6.6.tar.gz \
    && tar xvf v3.6.6.tar.gz && cd SEAL-3.6.6 \
    && cmake -S . -B build && cmake --build build -- -j$(nproc) && cmake --install build

WORKDIR /
# Install Eigen
RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz \
    && tar xvf eigen-3.4.0.tar.gz \
    && cp -r eigen-3.4.0/Eigen /usr/local/include/ \
    && cd eigen-3.4.0 && cmake -S . -B build

# Install LibTorch
RUN wget -O libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcpu.zip \
    && unzip libtorch.zip \
    && rm libtorch.zip

ENV LD_LIBRARY_PATH /libtorch/lib:/usr/local/lib:$LD_LIBRARY_PATH

COPY . /app
WORKDIR /app

RUN Eigen3_DIR=/eigen-3.4.0/build Torch_DIR=/libtorch/share/cmake/Torch \
    cmake -S . -B build && cmake --build build -- -j$(nproc)
