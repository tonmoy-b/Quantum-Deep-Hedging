# Step 1: Build Stage
FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y \
    build-essential cmake wget unzip libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install LibTorch for Linux in the image (CPU version for smaller image size)
# Todo: for GPU access in the container, use the 'cu126' download link instead
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.1.0%2Bcpu.zip \
    && unzip libtorch-shared-with-deps-2.1.0+cpu.zip -d /opt/ \
    && rm libtorch-shared-with-deps-2.1.0+cpu.zip

ENV CMAKE_PREFIX_PATH=/opt/libtorch

WORKDIR /app
COPY . .

# build
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release

# Runtime Stage 
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# copy the executable, pt model and the torch libraries
COPY --from=builder /app/build/hedging_engine .
COPY --from=builder /opt/libtorch/lib /usr/lib/
COPY ./models/saved.pt ./models/saved.pt

# expose the UDP port
EXPOSE 8080

CMD ["./hedging_engine", "models/saved.pt"]