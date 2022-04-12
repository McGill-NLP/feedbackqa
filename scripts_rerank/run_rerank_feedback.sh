#!/bin/sh


export MODEL_SAVE_PATH=''
export FB_MODEL_SAVE_PATH=''
export PRED_OUTPUT_PATH=''
export FB_OUTPUT_PATH=''

for DOMAIN in 'Australia' 'CDC' 'UK' 'WHO'
do
    python ./feedbackQA/rerank.py \
        --gpu 0 \
        --do_rerank \
        --predict ./feedbackQA_data/crowdsourced_data/$DOMAIN/test.json \
        --config ./feedbackQA/configs/bart_inve.yaml \
        --output $MODEL_SAVE_PATH \
        --reason_config ./feedbackQA/configs_rerank/feedback.yaml\
        --reason_output FB_MODEL_SAVE_PATH\
        --predict-to $PRED_OUTPUT_PATH/$DOMAIN_outs.txt\
        --feedback_output_file $FB_OUTPUT_PATH/$DOMAIN.json
done

python ./feedbackQA/rerank.py \
        --gpu 0 \
        --do_rerank \
        --predict ./feedbackQA_data/crowdsourced_data/Quebec/test_new_faq_coll4.json \
        --config ./feedbackQA/configs/bart_inve.yaml \
        --output $MODEL_SAVE_PATH \
        --reason_config ./feedbackQA/configs_rerank/feedback.yaml\
        --reason_output FB_MODEL_SAVE_PATH\
        --predict-to $PRED_OUTPUT_PATH/quebec_new_outs.txt\
        --feedback_output_file $FB_OUTPUT_PATH/quebec_new.json

python ./feedbackQA/rerank.py \
        --gpu 0 \
        --do_rerank \
        --predict ./feedbackQA_data/crowdsourced_data/Quebec/test_old_faq_coll4.json \
        --config ./feedbackQA/configs/bart_inve.yaml \
        --output $MODEL_SAVE_PATH \
        --reason_config ./feedbackQA/configs_rerank/feedback.yaml\
        --reason_output FB_MODEL_SAVE_PATH\
        --predict-to $PRED_OUTPUT_PATH/quebec_old_outs.txt\
        --feedback_output_file $FB_OUTPUT_PATH/quebec_old.json