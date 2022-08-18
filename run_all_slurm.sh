#!/bin/bash

#SBATCH --job-name=time_series_classification #Job name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fc55956@alunos.fc.ul.pt
#SBATCH --ntasks=2 # Run on n CPU
#SBATCH --gres=gpu:1
#SBATCH --mem=46G #Job memory request
#SBATCH --time=48:00:00 # Time limit hrs:min:sec
#SBATCH --output=./logs/%x_%j.log # Standard output and error log

./run_all.sh
