#!/bin/bash
#SBATCH --ntasks=4              # number of tasks
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12      # number of CPU cores per process
#SBATCH --partition=gpu         # put the job into the gpu partition/queue
#SBATCH --output=data/logs/job_%j.out   
#SBATCH --error=data/logs/job_%j.err
#SBATCH --mem=200G              # how much CPU memory can be allocated for the job (hardware limit: 384 GB per node)
#SBATCH --time=20:00:00         # maximum wall time allocated for the job (max 24h for the gpu partition)
#SBATCH --job-name=fed_client        # job name (default is the name of this file)
ml fosscuda/2019a 
PYTHON=/home/chattbap/anaconda3/bin/python
mpirun -np 1 $PYTHON Client/client.py --client_id=1 &
mpirun -np 1 $PYTHON Client/client.py --client_id=2 &
mpirun -np 1 $PYTHON Client/client.py --client_id=3 &
# mpirun -np 1 $PYTHON Client/client.py --client_id=4 &
# mpirun -np 1 $PYTHON Client/client.py --client_id=5 &
# mpirun -np 1 $PYTHON Client/client.py --client_id=6 &
# mpirun -np 1 $PYTHON Client/client.py --client_id=7 &
mpirun -np 1 $PYTHON Client/client.py --client_id=8

wait