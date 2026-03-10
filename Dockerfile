FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# 安装 C++ 核心编译管线：GCC, CMake, GDB 调试器, Valgrind 内存检测
RUN apt-get update && apt-get install -y \
    build-essential cmake gdb valgrind \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 让容器在后台休眠，保持存活
CMD ["tail", "-f", "/dev/null"]