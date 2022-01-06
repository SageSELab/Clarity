#!/bin/bash

x=281999
while [ $x -lt 500000 ];
do
    model_file="./models/low-captions-v3/$x.npy"
    echo $model_file
    output_file="./checkpoint_results/low-captions-v3-results/low-captions-check-$x-results.txt"
    echo $output_file
    python main.py --phase=eval --model_file=${model_file} --beam_size=3 > ${output_file}
    (( x+=1000 ))
done
exit 0

