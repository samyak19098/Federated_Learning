#!/bin/bash
#SBATCH --ntasks=2              # number of tasks
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4       # number of CPU cores per process
#SBATCH --partition=gpu         # put the job into the gpu partition/queue
#SBATCH --output=job_%j.out     # file name for stdout/stderr
#SBATCH --error=job_%j.err
#SBATCH --mem=200G              # how much CPU memory can be allocated for the job (hardware limit: 384 GB per node)
#SBATCH --time=00:00:30         # maximum wall time allocated for the job (max 24h for the gpu partition)
#SBATCH --job-name=test_job        # job name (default is the name of this file)

PYTHON=/home/chattbap/FederatedLearning_Framework/venv/bin/python
PROGRAM=/home/chattbap/FederatedLearning_Framework/extras/test_file_slurm.py

#srun echo ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS} 
srun $PYTHON $PROGRAM 
#srun $PYTHON $PROGRAM
srun $PYTHON $PROGRAM

'''
# !/bin/bash
# #SBATCH --ntasks=1               # number of tasks
# #SBATCH --ntasks-per-node=1     # processes per node
# SBATCH --nodes=1
# SBATCH --gres=gpu:2
# SBATCH --cpus-per-task=1       # number of CPU cores per process
# SBATCH --output=job_%j.out     # file name for stdout/stderr
# SBATCH --error=job_%j.err
# #SBATCH --mem=200G              # how much CPU memory can be allocated for the job (hardware limit: 384 GB per node)
# SBATCH --time=00:10:00         # maximum wall time allocated for the job (max 24h for the gpu partition)
# SBATCH --job-name=test_job        # job name (default is the name of this file)

#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # processes per node
#SBATCH --cpus-per-task=4       # number of CPU cores per process
#SBATCH --gres=gpu:4            # GPUs per node
##SBATCH --overcommit
#SBATCH --hint=compute_bound
#SBATCH --hint=multithread
##SBATCH --exclusive
#SBATCH --partition=gpu         # put the job into the gpu partition/queue
#SBATCH --output=job_%j.out     # file name for stdout/stderr
#SBATCH --error=job_%j.err
#SBATCH --mem=200G              # how much CPU memory can be allocated for the job (hardware limit: 384 GB per node)
#SBATCH --time=20:00:00         # maximum wall time allocated for the job (max 24h for the gpu partition)
#SBATCH --job-name=my_test        # job name (default is the name of this file)


# !/bin/bash
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCH --gres=gpu:1
# SBATCH --cpus-per-task=1      # number of CPU cores per process
# SBATCH --output=job_%j.out     # file name for stdout/stderr
# SBATCH --error=job_%j.err
# SBATCH --time=20:00:00         # maximum wall time allocated for the job (max 24h for the gpu partition)
# SBATCH --job-name=test_job        # job name (default is the name of this file)

#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2               # number of tasks
##SBATCH --ntasks-per-node=1     # processes per node
#SBATCH --ntasks-per-gpu=1
#SBATCH --gres=gpu:1
##SBATCH --gpus-per-task=volta:1
##SBATCH --gpu-bind=per_task:1
#SBATCH --cpus-per-task=4       # number of CPU cores per process
#SBATCH --partition=gpu         # put the job into the gpu partition/queue
#SBATCH --output=job_%j.out     # file name for stdout/stderr
#SBATCH --error=job_%j.err
#SBATCH --mem=200G              # how much CPU memory can be allocated for the job (hardware limit: 384 GB per node)
#SBATCH --time=20:00:00         # maximum wall time allocated for the job (max 24h for the gpu partition)
#SBATCH --job-name=test_job        # job name (default is the name of this file)
'''