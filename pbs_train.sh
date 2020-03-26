#!/bin/bash

#PBS -S /bin/bash
#PBS -N vamperouge
#PBS -j oe
#PBS -l walltime=08:00:00
#PBS -q gpuq -l select=1:ncpus=12:ngpus=1:gputype=p100:mem=100gb
#PBS -M corentin.dupret@student.ecp.fr
#PBS -m abe
#PBS -q gpuq
#PBS -P test

# Go to the current directory
cd $PBS_O_WORKDIR
[ ! -d output ] && mkdir output

# Module load 
module load anaconda3/5.3.1

# Activate anaconda environment code
source activate vamperouge

python main.py
