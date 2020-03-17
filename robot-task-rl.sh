#!/bin/bash
#SBATCH -p dev  # partition (queue)
#SBATCH -N 1  # number of nodes
#SBATCH -n 1  # number of tasks per node
#SBATCH -c 96  # number of cpus per task
#SBATCH --mem 744000  # memory pool for all cores (in megabytes)
#SBATCH -t 0-08:00  # time (D-HH:MM)
#SBATCH -o logs/slurm.%N.%j.out  # STDOUT
#SBATCH -e logs/slurm.%N.%j.err  # STDERR

/home/scheiderer/.conda/envs/robot_rl/bin/python trainer/trainer.py
