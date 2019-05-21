#!/bin/bash
#SBATCH -J Nembed-tr-co1
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
python train_word_embeddings.py --num-iterations 1200000 --path data/mscoco/
echo "ending"
