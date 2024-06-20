#!/bin/bash

# Define the ranges of values
pool_sizes=(10 100 1000)
max_iters=200
pull=1.0
push_values=(0.1 1.0 2.5 10.0)
x_0="Roger Federer is the "
rand_pool_options=(true false)

# Get the current date and time
datetime=$(date +%Y-%m-%d-%H%M%S)

# Loop through each combination of parameters
for pool_size in "${pool_sizes[@]}"
do
    for push in "${push_values[@]}"
    do
        for rand_pool in "${rand_pool_options[@]}"
        do
            # Create the output directory with the current datetime and unique identifying word
            output_dir="results/federer_pool${pool_size}_push${push}_randpool${rand_pool}_${datetime}"
            
            # Prepare the command with common parameters
            command="python3 scripts/greedy_forward_single.py \
                --model meta-llama/Meta-Llama-3-8B \
                --x_0 \"${x_0}\" \
                --output_dir \"${output_dir}\" \
                --max_iters ${max_iters} \
                --max_parallel 500 \
                --pool_size ${pool_size} \
                --push ${push} \
                --pull ${pull} \
                --frac_ext 0.2"
            
            # Add the --rand_pool argument if rand_pool is true
            if [ "${rand_pool}" = true ]; then
                command+=" --rand_pool"
            fi
            
            # Run the experiment with the current parameters
            eval "${command}"
        done
    done
done
