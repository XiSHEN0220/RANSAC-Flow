#!/bin/sh
# run jupyterlab from docker
###

echo "Usage: after running this script, open localhost:8888 in your host browser."

jupyter lab --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token=''
