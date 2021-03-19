#!/bin/bash

#SBATCH --partition=standard
#SBATCH --account=pi-dachxiu
#SBATCH --job-name=data_pre
#SBATCH --output=JOBLOG/Job_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --array=0-55

module load python36_anaconda/5.2.0
python -c "import get_mda; get_mda.get_mda(${SLURM_ARRAY_TASK_ID})"
