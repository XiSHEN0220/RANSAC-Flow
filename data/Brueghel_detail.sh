#!/bin/bash

SCRIPT_DIR="$(dirname $(realpath "$0"))"
(
    cd ${SCRIPT_DIR}
    if [ ! -d ArtMiner_Detail ]; then 
	wget https://www.dropbox.com/s/oqmqioi2jcqjxev/ArtMiner_Detail.zip?dl=0
	mv ArtMiner_Detail.zip?dl=0 ArtMiner_Detail.zip
	unzip ArtMiner_Detail.zip
	rm ArtMiner_Detail.zip
    fi

    if [ ! -d ArtMiner_Detail ]; then 
	echo "[ERROR] download data failed."
	exit 1
    fi
)
