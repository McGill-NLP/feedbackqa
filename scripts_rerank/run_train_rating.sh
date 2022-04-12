#!/bin/sh
export MODEL_SAVE_PATH=''
python ./feedbackQA/reason_ce.py --gpu 0,1,2 --train \
        --config ./feedbackQA/configs_rerank/rating.yaml\
        --output $MODEL_SAVE_PATH --rate_only