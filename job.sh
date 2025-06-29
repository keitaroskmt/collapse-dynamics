#!/bin/bash -l
#PBS -q regular-a
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -W group_list=gj24
#PBS -j oe

cd ${PBS_O_WORKDIR}

# execute your main program.
# for seed in 42 314 2718; do
#     uv run main.py num_steps=1000 embed_dim=2000 signal_norm=20 seed=$seed
# done e

seed=42
uv run train.py seed=$seed
uv run train.py optimizer=adamw seed=$seed
