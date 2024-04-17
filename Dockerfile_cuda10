# for cuda compute capability < 8 GPU.
FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime
WORKDIR /app
COPY . .
# to avoid the error "libgthread-2.0.so.0: cannot open", we need opencv itself.
RUN apt-get update && \
    apt-get install -y libopencv-dev
RUN bash requirement.sh