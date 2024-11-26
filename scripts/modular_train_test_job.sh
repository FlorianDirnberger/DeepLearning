#!/bin/sh
#BSUB -q gpua100
#BSUB -J PREPRO1
### number of core
#BSUB -n 4
### specify that all cores should be on the same host
#BSUB -gpu "num=1:mode=exclusive_process"
### specify the memory needed
#BSUB -R "rusage[mem=10GB]"
### -- send notification at completion -- 
#BSUB -N 
### Number of hours needed
#BSUB -W 23:59
### added outputs and errors to files
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err

echo "Running script..."

module load cuda/11.8
module load python3/3.10.13
source ~/my_project_dir/02456_grp_99_venv/bin/activate

# Run the Python script
python3 ~/my_project_dir/DeepLearning97/scripts/modular_train_test.py > log/modular_train_test$(date +"%d-%m-%y")_$(date +'%H:%M:%S').log
