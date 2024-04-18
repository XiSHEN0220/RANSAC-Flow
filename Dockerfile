# for cuda compute capability >= 8 GPU.
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
WORKDIR /app
COPY . .

# tzdata to avoid stopping at being asked Geographic area during libopencv-dev installation
RUN apt-get update && \
    apt-get install -y tzdata wget unzip
# to avoid the error "libgthread-2.0.so.0: cannot open", we need opencv itself.
RUN apt-get install -y libopencv-dev

RUN pip install -r requirements.txt
