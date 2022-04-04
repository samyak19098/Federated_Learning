#!/bin/bash
#SBATCH --nodes=4               # number of nodes
#SBATCH --ntasks-per-node=4     # processes per node
#SBATCH --cpus-per-task=18       # number of CPU cores per process
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

##ml fosscuda/2019a
# export OMP_NUM_THREADS=72

##PYTHON=/home/chattbap/anaconda3/bin/python
PYTHON=/home/chattbap/FederatedLearning_Framework/venv/bin/python
PROGRAM=/home/chattbap/FederatedLearning_Framework/start_reducer.py
DATADIR=/home/chattbap/workspace/data
IMAGENETDIR=/home/chattbap/ILSVRC
MASTER=`/bin/hostname -s`
MASTERIP=`getent hosts $MASTER | cut -d' ' -f1`

srun \
$PYTHON $PROGRAM  --data-dir $DATADIR  --imagenet-dir $IMAGENETDIR \
--dataset imagenet --num-classes 1000 --momentum 0.9 --model res50 \
--cuda --train_processing_bs 32 --test_processing_bs 32 --lr 0.1 \
--baseline_lr 0.0125 --weight-decay 0.00005 --seed 42 --pm \
--nesterov --workers 8 --num-threads 8 --test-freq 1 --partition \
--num-processes 4 --pre_post_epochs 45 \
--scheduler-type mstep --lrmilestone 30 60 80 \
--av_type a2a --training-type mnalsgHW \
--bs_multiple 1 --test_bs_multiple 1 --epochs 90 --averaging_freq 8 \
--warm_up_epochs 5 --dist-url tcp://$MASTER:23456  \
--numnodes $SLURM_JOB_NUM_NODES --storeresults

# mpirun -n 8 -bind-to none -map-by slot \
# srun \
# $PYTHON $PROGRAM  --data-dir $DATADIR  --imagenet-dir $IMAGENETDIR \
# --dataset imagenet --num-classes 1000 --momentum 0.9 --model res50 \
# --cuda --train_processing_bs 128 --test_processing_bs 128 --lr 0.4 \
# --baseline_lr 0.0125 --weight-decay 0.00005 --seed 0 --pm  \
# --nesterov --workers 8 --num-threads 8 --test-freq 1 --partition \
# --scheduler-type mstep --lrmilestone 30 60 80 --pre_post_epochs 45 \
# --av_type a2a --training-type mnddp --numnodes $SLURM_JOB_NUM_NODES \
# --bs_multiple 1 --test_bs_multiple 1 --epochs 90 \
# --warm_up_epochs 5 --dist-url tcp://$HOSTNAME:23456  --storeresults


# mpirun -n 8 -bind-to none -map-by slot \
# srun \
# $PYTHON $PROGRAM  --data-dir $DATADIR  --imagenet-dir $IMAGENETDIR \
# --dataset imagenet --num-classes 1000 --momentum 0.9 --model res50 \
# --cuda --train_processing_bs 128 --test_processing_bs 128 --lr 0.4 \
# --baseline_lr 0.0125 --weight-decay 0.00005 --seed 0 --pm  \
# --nesterov --workers 8 --num-threads 8 --test-freq 1 --partition \
# --scheduler-type mstep --lrmilestone 30 60 80 --pre_post_epochs 45 \
# --averaging_freq 8 --numnodes $SLURM_JOB_NUM_NODES \
# --av_type a2a --training-type mnlSGD --pre_post_epochs 45 \
# --bs_multiple 1 --test_bs_multiple 1 --epochs 90 \
# --warm_up_epochs 5 --dist-url tcp://$MASTER:21456  --storeresults



# mpirun -n 8 -bind-to none -map-by slot \
# srun \
# $PYTHON $PROGRAM  --data-dir $DATADIR  --imagenet-dir $IMAGENETDIR \
# --dataset imagenet --num-classes 1000 --momentum 0.9 --model res50 \
# --cuda --train_processing_bs 32 --test_processing_bs 32 --lr 0.125 \
# --baseline_lr 0.0125 --weight-decay 0.00005 --seed 42 --pm \
# --nesterov --workers 8 --num-threads 8 --test-freq 1 --partition \
# --num-processes 4 --pre_post_epochs 45 \
# --scheduler-type mstep --lrmilestone 30 60 80 \
# --av_type a2a --training-type mnalsgPHW  --prepassmepochs 6 \
# --bs_multiple 1 --test_bs_multiple 1 --epochs 90 --averaging_freq 8 \
# --warm_up_epochs 5 --dist-url tcp://$MASTER:23456  \
# --numnodes $SLURM_JOB_NUM_NODES --storeresults

#srun \
#$PYTHON $PROGRAM  --data-dir $DATADIR  --imagenet-dir $IMAGENETDIR \
#--dataset imagenet --num-classes 1000 --momentum 0.9 --model res50 \
#--cuda --train_processing_bs 32 --test_processing_bs 32 --lr 0.1 \
#--baseline_lr 0.0125 --weight-decay 0.00005 --seed 42 --pm \
#--nesterov --workers 8 --num-threads 8 --test-freq 1 --partition \
#--num-processes 4 --pre_post_epochs 45 \
#--scheduler-type mstep --lrmilestone 30 60 80 \
#--av_type a2a --training-type mnalsgHW \
#--bs_multiple 1 --test_bs_multiple 1 --epochs 90 --averaging_freq 8 \
#--warm_up_epochs 5 --dist-url tcp://$MASTER:23456  \
#--numnodes $SLURM_JOB_NUM_NODES --storeresults



# MASTERIP=`ping -c1 -t1 -W0 $MASTER`# | tr -d '():' | awk '/^PING/{print $3}'`
# SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`

# mpirun -n 8 -bind-to none -map-by slot \
# echo "MASTER=" $MASTER $MASTERIP #$SLURM_JOB_NODELIST "SLAVES=" && $SLAVES /bin/hostname -s


# mpirun -n 8 -bind-to none -map-by slot \
# $PYTHON $PROGRAM \
# --dataset cifar10 --num-classes 10 --momentum 0.9 --model res20 \
# --cuda --train_processing_bs 1024 --test_processing_bs 1024 --lr 0.8 \
# --baseline_lr 0.1 --weight-decay 0.0005 --seed 0 --pm --averaging_freq 8 \
# --nesterov --workers 4 --num-threads 4 --test-freq 1 --partition \
# --scheduler-type mstep --training-type mnddp --data-dir $DATADIR \
# --bs_multiple 1 --test_bs_multiple 1 --epochs 2 --warm_up_epochs \
# 5 --dist-url tcp://$MASTER:23456 --lrmilestone 150 225  --storeresults


# mpirun -n 8 -bind-to none -map-by slot \
# $PYTHON $PROGRAM  \
# --dataset imagenet --num-classes 1000 --momentum 0.9 --model res18 \
# --cuda --train_processing_bs 256 --test_processing_bs 256 --lr 0.8 \
# --baseline_lr 0.01 --weight-decay 0.00005 --seed 42 --pm  \
# --nesterov --workers 4 --num-threads 4 --test-freq 1 --partition \
# --scheduler-type cosine \
# --av_type a2a --training-type mnddp \
# --bs_multiple 1 --test_bs_multiple 1 --epochs 5 --imagenet-dir \
# $IMAGENETDIR --warm_up_epochs 5 \
# --dist-url tcp://$MASTER:23456  --storeresults


# mpirun -n 4 -bind-to none -map-by slot \
# $PYTHON $PROGRAM  \
# --dataset imagenet --num-classes 1000 --momentum 0.9 --model res18 \
# --cuda --train_processing_bs 32 --test_processing_bs 32 --lr 0.1 \
# --baseline_lr 0.01 --weight-decay 0.00005 --seed 42 --pm  \
# --nesterov --workers 4 --num-threads 4 --test-freq 1 --partition \
# --num-processes 4 --pre_post_epochs 0 --scheduler-type cosine \
# --av_type a2a --training-type mnalsgHW  --prepassmepochs 5 \
# --bs_multiple 1 --test_bs_multiple 1 --epochs 1 --imagenet-dir \
# $IMAGENETDIR --averaging_freq 16 --warm_up_epochs 5 \
# --dist-url tcp://$MASTER:23456  --storeresults


# mpirun -n 8 -bind-to none -map-by slot \
# $PYTHON $PROGRAM  --data-dir $DATADIR  --imagenet-dir $IMAGENETDIR \
# --dataset imagenet --num-classes 1000 --momentum 0.9 --model res50 \
# --cuda --train_processing_bs 32 --test_processing_bs 32 --lr 0.1 \
# --baseline_lr 0.0125 --weight-decay 0.00005 --seed 0 --pm  \
# --nesterov --workers 4 --num-threads 4 --test-freq 1 --partition \
# --num-processes 4 --scheduler-type cosine \
# --av_type a2a --training-type mnalsgHW \
# --bs_multiple 1 --test_bs_multiple 1 --epochs 3 --averaging_freq 16 \
# --warm_up_epochs 5 --dist-url tcp://$MASTER:23456  --storeresults

# mpirun -n 8 -bind-to none -map-by slot \
# $PYTHON $PROGRAM  --data-dir $DATADIR  --imagenet-dir $IMAGENETDIR \
# --dataset imagenet --num-classes 1000 --momentum 0.9 --model res50 \
# --cuda --train_processing_bs 128 --test_processing_bs 128 --lr 0.4 \
# --baseline_lr 0.0125 --weight-decay 0.00005 --seed 0 --pm  \
# --nesterov --workers 4 --num-threads 4 --test-freq 1 --partition \
# --scheduler-type cosine \
# --av_type a2a --training-type mnddp \
# --bs_multiple 1 --test_bs_multiple 1 --epochs 90 \
# --warm_up_epochs 5 --dist-url tcp://$MASTER:23456  --storeresults

# mpirun -n 8 -bind-to none -map-by slot \
# $PYTHON $PROGRAM  --data-dir $DATADIR  --imagenet-dir $IMAGENETDIR \
# --dataset imagenet --num-classes 1000 --momentum 0.9 --model res50 \
# --cuda --train_processing_bs 32 --test_processing_bs 32 --lr 0.125 \
# --baseline_lr 0.0125 --weight-decay 0.00005 --seed 0 --pm  \
# --nesterov --workers 4 --num-threads 4 --test-freq 1 --partition \
# --num-processes 4 --pre_post_epochs 45 --scheduler-type cosine \
# --av_type a2a --training-type mnalsgPHW  --prepassmepochs 9 \
# --bs_multiple 1 --test_bs_multiple 1 --epochs 90 --averaging_freq 16 \
# --warm_up_epochs 5 --dist-url tcp://$MASTER:23456  --storeresults

# mpirun -n 8 -bind-to none -map-by slot \
# $PYTHON $PROGRAM  --data-dir $DATADIR  --imagenet-dir $IMAGENETDIR \
# --dataset imagenet --num-classes 1000 --momentum 0.9 --model res50 \
# --cuda --train_processing_bs 128 --test_processing_bs 128 --lr 0.4 \
# --baseline_lr 0.0125 --weight-decay 0.00005 --seed 0 --pm  \
# --nesterov --workers 4 --num-threads 6 --test-freq 1 --partition \
# --scheduler-type mstep --lrmilestone 30 60 80 --pre_post_epochs 45 \
# --av_type a2a --training-type mnlSGD --averaging_freq 8 \
# --bs_multiple 1 --test_bs_multiple 1 --epochs 90 \
# --warm_up_epochs 5 --dist-url tcp://$MASTER:23456  --storeresults


# mpirun -n 8 -bind-to none -map-by slot \
# $PYTHON $PROGRAM  --data-dir $DATADIR  --imagenet-dir $IMAGENETDIR \
# --dataset imagenet --num-classes 1000 --momentum 0.9 --model res50 \
# --cuda --train_processing_bs 32 --test_processing_bs 32 --lr 0.1 \
# --baseline_lr 0.0125 --weight-decay 0.00005 --seed 0 --pm  \
# --nesterov --workers 8 --num-threads 8 --test-freq 1 --partition \
# --num-processes 3 --pre_post_epochs 45 --scheduler-type cosine \
# --av_type a2a --training-type mnalsgHW \
# --bs_multiple 1 --test_bs_multiple 1 --epochs 90 --averaging_freq 16 \
# --warm_up_epochs 5 --dist-url tcp://$MASTER:23456  --storeresults
