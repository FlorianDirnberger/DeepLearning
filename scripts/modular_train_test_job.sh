#!/bin/sh
#BSUB -q gpua100
#BSUB -J Sweep_hopeful826_ConvLayers_GRID
### number of core
#BSUB -n 4
### specify that all cores should be on the same host
#BSUB -gpu "num=1:mode=exclusive_process"
### specify the memory needed
#BSUB -R "rusage[mem=10GB]"
### Number of hours needed
#BSUB -W 71:59
### added outputs and errors to files
#BSUB -o outputs/Output_%J.out
#BSUB -e outputs/Error_%J.err

echo "Running script..."

module load cuda/11.8
module load python3/3.10.13
source ~/DL/p4_velocity/DL_97_venv/bin/activate
python3 /dtu/blackhole/13/212723/DeepLearning97/scripts/modular_train_test.py > log/modular_train_test$(date +"%d-%m-%y")_$(date +'%H:%M:%S').log
