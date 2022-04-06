cd 
cp /home/lcc/scratch/bart-base/ $SLURM_TMPDIR/workspace/bart-base/ -r
python /home/lcc/rqa_feedback/code_base/feedbackQA/main_tune.py --gpu 0 --train \
    --config /home/lcc/rqa_feedback/code_base/feedbackQA/configs/bart_inve.yaml \
    --output output_$SLURM_JOB_ID/
mkdir ~/scratch/feedbackQA_models/
mkdir ~/scratch/feedbackQA_models/bart_qa/
cp -a ./output_$SLURM_JOB_ID/ ~/scratch/feedbackQA_models/bart_qa/