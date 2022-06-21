FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime

WORKDIR /app
COPY . .

RUN bash requirement.sh
RUN cd /app/model/pretrained && \
    bash download_model.sh
RUN cd /app/data && \
    bash Brueghel_detail.sh # Brueghel detail dataset (208M) : visual results, aligning groups of details
