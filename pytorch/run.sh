#!/bin/bash
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by - Srikanth Madikeri (2017)
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

python main-5cnn2fc-768-400.py \
    --no-cuda --epochs=41 --lr=0.0001 \
    --batch-size=10 --test-batch-size=10 \
    --output-dir model-output/5cnn-2fc-mfm-768x400 


# To check final EER, use Kaldi's compute-eer
compute-eer eer-file
