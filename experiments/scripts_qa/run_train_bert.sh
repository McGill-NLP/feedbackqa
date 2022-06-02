#!/bin/sh
export MODEL_SAVE_PATH=''

python ./feedbackQA/main_tune.py --gpu 0,1 --train \
    --config ./feedbackQA/configs/bert_inve.yaml \
    --output $MODEL_SAVE_PATH