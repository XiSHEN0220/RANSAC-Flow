FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime

WORKDIR /app
COPY . .

# to avoid the error "libgthread-2.0.so.0: cannot open", we need opencv itself.
RUN apt-get update && apt-get install -y libopencv-dev
RUN apt-get install -y wget
RUN bash requirement.sh
RUN cd /app/model/pretrained && \
    bash download_model.sh
RUN cd /app/data && \
    bash Brueghel_detail.sh # Brueghel detail dataset (208M) : visual results, aligning groups of details
