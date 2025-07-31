#!/bin/bash
#SBATCH -J train_dynedge
#SBATCH -c 4 # Number of cores
#SBATCH -p arguelles_delgado_gpu
#SBATCH --mem 8000
#SBATCH -t 0-4:00:00 # Maximum execution time (D-HH:MM:SS)
#SBATCH -o /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/double_cascade_identification/slurm_logs/out/dynedge_energy%A.out # Standard output
#SBATCH -e /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/double_cascade_identification/slurm_logs/err/dynedge_energy%A.err # Standard error
#SBATCH --gres gpu:nvidia_a100-sxm4-80gb:1
source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/spack/share/spack/setup-env.sh
spack env activate graphnet
module load cuda/11.8.0-fasrc01
CMD="python /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jbook/double_cascade_identification/scripts/test_estimator.py" 
echo $CMD
$CMD
