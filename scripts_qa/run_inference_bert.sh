#!/bin/sh
#SBATCH --job-name=qa_inference
#SBATCH --account=def-bengioy
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --signal=SIGINT
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH --output=/home/lcc/scratch/logs/qa_predict/%j.out
#SBATCH --error=/home/lcc/scratch/logs/qa_predict/%j.error
source /home/lcc/rqa_feedback/code_base/feedbackQA/cc_job_new.sh
cd $SLURM_TMPDIR/workspace/parlai/
python setup.py install
export MODEL_OUT_PATH='/home/lcc/rqa_feedback/data/model_preds/bart_qa_run1/'
mkdir $MODEL_OUT_PATH
cp -a $HOME/rqa_feedback/code_base/bert_reranker/ $SLURM_TMPDIR/workspace/ -r
mkdir $SLURM_TMPDIR/workspace/qa_model/
mkdir $SLURM_TMPDIR/workspace/qa_model/output11/
cp /home/lcc/scratch/from_prakhar/best_checkpoints/bart/multidomain/inve/output_4/*.ckpt \
    $SLURM_TMPDIR/workspace/qa_model/output11/ -r
ls $SLURM_TMPDIR/workspace/qa_model/output11/
cd 
python /home/lcc/rqa_feedback/code_base/feedbackQA/qa_inference.py \
        --gpu 0 \
        --predict /home/lcc/rqa_feedback/data/crowdsourced_data/Australia/test.json \
        --config /home/lcc/rqa_feedback/code_base/feedbackQA/configs/bert_inve.yaml \
        --output /home/lcc/scratch/feedbackQA_models/bert_qa/output_28475281 \
        --predict-to $SLURM_TMPDIR/workspace/australia_outs.txt\
        --feedback_output_file $SLURM_TMPDIR/workspace/feedback_australia.json
cp $SLURM_TMPDIR/workspace/australia_outs.txt $MODEL_OUT_PATH/australia_outs.txt

python /home/lcc/rqa_feedback/code_base/feedbackQA/qa_inference.py \
        --gpu 0 \
        --predict /home/lcc/rqa_feedback/data/crowdsourced_data/CDC/test.json \
        --config /home/lcc/rqa_feedback/code_base/feedbackQA/configs/bert_inve.yaml \
        --output /home/lcc/scratch/feedbackQA_models/bert_qa/output_28475281 \
        --predict-to $SLURM_TMPDIR/workspace/cdc_outs.txt\
        --feedback_output_file $SLURM_TMPDIR/workspace/feedback_cdc.json
cp $SLURM_TMPDIR/workspace/cdc_outs.txt $MODEL_OUT_PATH/cdc_outs.txt

python /home/lcc/rqa_feedback/code_base/feedbackQA/qa_inference.py \
        --gpu 0 \
        --predict /home/lcc/rqa_feedback/data/crowdsourced_data/UK/test.json \
        --config /home/lcc/rqa_feedback/code_base/feedbackQA/configs/bert_inve.yaml \
        --output /home/lcc/scratch/feedbackQA_models/bert_qa/output_28475281 \
        --predict-to $SLURM_TMPDIR/workspace/uk_outs.txt\
        --feedback_output_file $SLURM_TMPDIR/workspace/feedback_uk.json
cp $SLURM_TMPDIR/workspace/uk_outs.txt $MODEL_OUT_PATH/uk_outs.txt

python /home/lcc/rqa_feedback/code_base/feedbackQA/qa_inference.py \
        --gpu 0 \
        --predict /home/lcc/rqa_feedback/data/crowdsourced_data/WHO/test.json \
        --config /home/lcc/rqa_feedback/code_base/feedbackQA/configs/bert_inve.yaml \
        --output /home/lcc/scratch/feedbackQA_models/bert_qa/output_28475281 \
        --predict-to $SLURM_TMPDIR/workspace/who_outs.txt\
        --feedback_output_file $SLURM_TMPDIR/workspace/feedback_who.json
cp $SLURM_TMPDIR/workspace/who_outs.txt $MODEL_OUT_PATH/who_outs.txt

python /home/lcc/rqa_feedback/code_base/feedbackQA/qa_inference.py \
        --gpu 0 \
        --predict /home/lcc/rqa_feedback/data/crowdsourced_data/Quebec/test_new_faq_coll4.json \
        --config /home/lcc/rqa_feedback/code_base/feedbackQA/configs/bert_inve.yaml \
        --output /home/lcc/scratch/feedbackQA_models/bert_qa/output_28475281 \
        --predict-to $SLURM_TMPDIR/workspace/quebec_new_outs.txt\
        --feedback_output_file $SLURM_TMPDIR/workspace/feedback_who.json
cp $SLURM_TMPDIR/workspace/quebec_new_outs.txt $MODEL_OUT_PATH/quebec_new_outs.txt


python /home/lcc/rqa_feedback/code_base/feedbackQA/qa_inference.py \
        --gpu 0 \
        --predict /home/lcc/rqa_feedback/data/crowdsourced_data/Quebec/test_old_faq_coll4.json \
        --config /home/lcc/rqa_feedback/code_base/feedbackQA/configs/bert_inve.yaml \
        --output /home/lcc/scratch/feedbackQA_models/bert_qa/output_28475281 \
        --predict-to $SLURM_TMPDIR/workspace/quebec_old_outs.txt\
        --feedback_output_file $SLURM_TMPDIR/workspace/feedback_who.json
cp $SLURM_TMPDIR/workspace/quebec_old_outs.txt $MODEL_OUT_PATH/quebec_old_outs.txt