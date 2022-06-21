#!/bin/bash

if [ ! -d ArtMiner_Detail ]; then 
    wget https://www.dropbox.com/s/oqmqioi2jcqjxev/ArtMiner_Detail.zip?dl=0
    mv ArtMiner_Detail.zip?dl=0 ArtMiner_Detail.zip
    unzip ArtMiner_Detail.zip
    rm ArtMiner_Detail.zip
fi
