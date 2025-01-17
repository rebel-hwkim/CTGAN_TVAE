#!/bin/bash
BATCH=(1 2 4 8 16 32 64 128 256 512 1024)

for i in "${BATCH[@]}"
do
    CUDA_VISIBLE_DEVICES=3 python3 benchmark_gpu.py --batch_size $i --repeat 100
done
