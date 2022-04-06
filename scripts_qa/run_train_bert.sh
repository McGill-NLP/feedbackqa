#!/bin/sh
#SBATCH --job-name=bert_inve_poly
#SBATCH --account=rrg-bengioy-ad
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --signal=SIGINT
#SBATCH --gres=gpu:v100l:4
#SBATCH --mem=32000M
#SBATCH --cpus-per-task=24
#SBATCH --output=/home/lcc/scratch/logs/bert_inve_qa%j.out
#SBATCH --error=/home/lcc/scratch/logs/bert_inve_qa%j.error
source /home/lcc/rqa_feedback/code_base/feedbackQA/cc_job_new.sh
cd $SLURM_TMPDIR/workspace/parlai/
python setup.py install
cp /home/lcc/scratch/bart-base/ $SLURM_TMPDIR/workspace/bart-base/ -r
python /home/lcc/rqa_feedback/code_base/feedbackQA/main_tune.py --gpu 0,1 --train \
       --config /home/lcc/rqa_feedback/code_base/feedbackQA/configs/bert_inve.yaml \
       --output output_$SLURM_JOB_ID/
mkdir ~/scratch/feedbackQA_models/
mkdir ~/scratch/feedbackQA_models/bert_qa/
cp -a ./output_$SLURM_JOB_ID/ ~/scratch/feedbackQA_models/bert_qa/