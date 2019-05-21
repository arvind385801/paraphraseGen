#!/bin/bash
#SBATCH -J tst-gpu
# Partition to use - this example is commented out
#SBATCH -p gpucompute
# Pick nodes with feature 'foo'. Different clusters have 
# different features available
# but most of the time you don't need this
# next is for specified constraints of features, p100 is for envidia
# Memory
# GPU 
#SBATCH --gres=gpu:1
module load cuda91/toolkit/9.1.85
echo "starting own vrnn"
source activate paraGen3
python test.py --path data/quora/ --save-model data/quora/19-04-20_17-59/trained_RVAE
echo "ending"
