#!/bin/bash --login
#SBATCH --job-name=train
#SBATCH -o slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH -G 4 
#SBATCH --mem=64G
#SBATCH -w ilab3

cd ~/dev/CNN-Geoguesser
source .venv/bin/activate

python3 src/train.py
