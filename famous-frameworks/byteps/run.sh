#!/bin/bash

path="`dirname $0`"

export PATH=~/.local/bin:$PATH
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=1234

echo "Launch scheduler"
export DMLC_ROLE=scheduler
bpslaunch &

echo "Launch server"
export DMLC_ROLE=server
bpslaunch &

export NVIDIA_VISIBLE_DEVICES=0,1,2,3
export DMLC_WORKER_ID=0
export DMLC_ROLE=worker

bpslaunch python tensorflow2_mnist.py