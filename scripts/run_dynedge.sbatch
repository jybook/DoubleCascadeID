#!/bin/bash
#SBATCH -J train_dynedge
#SBATCH -c 4 # Number of cores
#SBATCH -p arguelles_delgado_gpu
#SBATCH --mem 12000
#SBATCH -t 1-04:00:00 # Maximum execution time (D-HH:MM:SS)
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/double_cascade_identification/slurm_logs/out/dynedge_energy%A.out # Standard output
#SBATCH -e /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/double_cascade_identification/slurm_logs/err/dynedge_energy%A.err # Standard error

source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/spack/share/spack/setup-env.sh
spack env activate graphnet
module load cuda/11.8.0-fasrc01
CMD="python /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/double_cascade_identification/scripts/01_train_dynedge.py" 
echo $CMD
$CMD