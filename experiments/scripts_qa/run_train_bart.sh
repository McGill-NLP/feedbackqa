#!/bin/sh

export MODEL_SAVE_PATH=''

python ./feedbackQA/main_tune.py --gpu 0,1 --train \
    --config ./feedbackQA/configs/bart_inve.yaml \
    --output $MODEL_SAVE_PATH