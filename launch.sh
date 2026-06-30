#!/bin/bash
#SBATCH -A <your-project-account>
#SBATCH --output=results/logs/%j.out
#SBATCH --error=results/logs/%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 3-00:00:00 

# module load Miniforge3/24.7.1-2-hpc1-bdist
# mamba activate pytorch-2.6.0

learning_rates=(0.01 0.001 0.0001)
batch_sizes=(32 64 128)
hidden_dims=(16 32 64)

for lr in "${learning_rates[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for hd in "${hidden_dims[@]}"; do
      for depth in "${depths[@]}"; do

        srun python train.py \
          --lr "${lr}" \
          --batch_size "${bs}" \
          --hidden_dim "${hd}" \
          --depth "${depth}"

      done
    done
  done
done
