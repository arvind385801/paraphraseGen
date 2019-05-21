#!/bin/bash
#SBATCH -J tstCoco-gpu
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
python test.py --path data/mscoco/ --save-model data/mscoco/19-04-26_22-22/trained_RVAE
echo "ending"
