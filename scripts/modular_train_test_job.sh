#!/bin/sh
#BSUB -q gpua100
#BSUB -J sweep_basic
### number of core
#BSUB -n 4
### specify that all cores should be on the same host
#BSUB -gpu "num=1:mode=exclusive_process"
### specify the memory needed
#BSUB -R "rusage[mem=10GB]"
### Number of hours needed
#BSUB -W 100:00
### added outputs and errors to files
#BSUB -o outputs/Output_%J.out
#BSUB -e outputs/Error_%J.err

echo "Running script..."

module load cuda/11.8
module load python3/3.10.13
source ~02456_grp_97_venv/bin/activate # Update this path to reflect your venv name
python3 ~zhome/9e/8/212358/Documents/DeepLearning/scripts/modular_train_test_rnn.py > log/modular_train_test_rnn$(date +"%d-%m-%y")_$(date +'%H:%M:%S').log
