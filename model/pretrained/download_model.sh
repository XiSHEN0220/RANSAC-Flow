#!/bin/bash

# params
EXPECT_NUM_MODELS=8

# process
SCRIPT_DIR="$(dirname $(realpath "$0"))"
(
    cd ${SCRIPT_DIR}

    models=(*.pth)
    NUM_DOWNLOADED_MODELS=${#models[@]}
    if [ ${NUM_DOWNLOADED_MODELS} -lt ${EXPECT_NUM_MODELS} ]; then
	wget https://www.dropbox.com/s/uegv8aqq5sj3542/model.zip?dl=0
	mv model.zip?dl=0 model.zip
	unzip model.zip
	rm model.zip
    fi

    # check whether the download is succeeded or not.
    models=(*.pth)
    NUM_DOWNLOADED_MODELS=${#models[@]}
    if [ ${NUM_DOWNLOADED_MODELS} -lt ${EXPECT_NUM_MODELS} ]; then
	echo "[ERROR] number of downloaded models = ${NUM_DOWNLOADED_MODELS}, but expected ${EXPECT_NUM_MODELS}."
	exit 1
    fi
)
