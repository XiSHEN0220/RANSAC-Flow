#!/bin/bash

EXPECT_NUM_MODELS=8

models=( *.pth )
NUM_DOWNLOADED_MODELS=${#models[@]}

if [ ${NUM_DOWNLOADED_MODELS} -lt ${EXPECT_NUM_MODELS} ]; then
    wget https://www.dropbox.com/s/uegv8aqq5sj3542/model.zip?dl=0
    mv model.zip?dl=0 model.zip
    unzip model.zip
    rm model.zip
fi
