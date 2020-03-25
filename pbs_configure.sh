#!/bin/bash

#PBS -S /bin/bash
#PBS -N vamperouge
#PBS -j oe
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=1
#PBS -q gpuq 
#PBS -P test

# Go to the directory where the job has been submitted 
cd $PBS_O_WORKDIR

# Setup conda env - ensure your .conda dir is located on your workir, and move it if not
[ -L ~/.conda ] && unlink ~/.conda
[ -d ~/.conda ] && mv -v ~/.conda $WORKDIR
[ ! -d $WORKDIR/.conda ] && mkdir $WORKDIR/.conda
ln -s $WORKDIR/.conda ~/.conda

# Module load
module load anaconda3/5.3.1

# Create conda environment
conda env create -f config/environment.yml --force

# Save environment description
source activate vamperouge
bash config/config_env.sh
conda env export > config/environment.yml
