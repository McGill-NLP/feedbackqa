#!/bin/sh
#SBATCH --job-name=bart_qa
#SBATCH --account=rrg-bengioy-ad
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --signal=SIGINT
#SBATCH --gres=gpu:v100l:2
#SBATCH --mem=32000M
#SBATCH --cpus-per-task=24
#SBATCH --output=/home/lcc/scratch/logs/bart_inve_qa%j.out
#SBATCH --error=/home/lcc/scratch/logs/bart_inve_qa%j.error
source /home/lcc/rqa_feedback/code_base/feedbackQA/cc_job_new.sh
cd $SLURM_TMPDIR/workspace/parlai/
python setup.py install
cd 
cp /home/lcc/scratch/bart-base/ $SLURM_TMPDIR/workspace/bart-base/ -r
python /home/lcc/rqa_feedback/code_base/feedbackQA/main_tune.py --gpu 0,1 --train \
    --config /home/lcc/rqa_feedback/code_base/feedbackQA/configs/bart_inve.yaml \
    --output $SLURM_TMPDIR/workspace/output_$SLURM_JOB_ID/
mkdir ~/scratch/feedbackQA_models/
mkdir ~/scratch/feedbackQA_models/bart_qa/
cp -a $SLURM_TMPDIR/workspace/output_$SLURM_JOB_ID/ ~/scratch/feedbackQA_models/bart_qa/