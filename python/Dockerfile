# ==========================================
# build --> c++ inference 
# ==========================================
FROM ubuntu:22.04 AS builder

# install build tools
RUN apt-get update && apt-get install -y build-essential cmake wget unzip libgomp1

WORKDIR /build

RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.1.0%2Bcpu.zip \
    && unzip libtorch-shared-with-deps-2.1.0+cpu.zip -d /opt/

# Copy C++ source from the root context
COPY ./HedgingEngine ./HedgingEngine
WORKDIR /build/HedgingEngine
RUN cmake -B build -DCMAKE_PREFIX_PATH=/opt/libtorch -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release

# ==========================================
# Runtime --> The "QaaS" Microservice
# ==========================================
FROM python:3.10-slim

# Install runtime dependencies for Kafka and C++ binary
RUN apt-get update && apt-get install -y libgomp1 librdkafka-dev gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install py requirements-pinned.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. copy c++ executable from build stage
COPY --from=builder /build/HedgingEngine/build/quantum_inference_engine /usr/local/bin/

# copy legend and py code
COPY ./legend ./legend
COPY ./python ./python

# environement vars
ENV PROJECT_TYPE="PROTOTYPE"
ENV DEFAULT_BRANCH="main"

ENTRYPOINT ["python", "python/train.py"]