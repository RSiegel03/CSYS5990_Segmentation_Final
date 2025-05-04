#!/bin/bash
#SBATCH --partition=week
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=40G
#SBATCH --job-name=log_reg_train
#SBATCH --time=130:00:00
#SBATCH --output=log_reg_train.out
#SBATCH --account=cs3540

source ~/.bashrc
# echo each command to the log file
set -x
# MLM3 has LGBM and required tools
conda activate MLFin_env
python3.13 train_model_vacc.py


