#!/bin/sh
#SBATCH --nodes=1              # number of tasks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --output=data/logs/job_%j.out   
#SBATCH --error=data/logs/job_%j.err
#SBATCH --mem=100G              # how much CPU memory can be allocated for the job (hardware limit: 384 GB per node)
#SBATCH --time=20:00:00         # maximum wall time allocated for the job (max 24h for the gpu partition)
#SBATCH --job-name=fed_reducer        # job name (default is the name of this file)

ml fosscuda/2019a 
PYTHON=/home/chattbap/anaconda3/bin/python

mpirun $PYTHON start_reducer.py

