# Controllability Results 

This directory (`results/controllability/`) is where we will store all the results on controlling language models toward **random** final token generation. 

Experiments will be largely similar to the reachability experiments (in `results/reachability`) performed in September -- except that a random token will be selected for `answer_ids`. 


## Falcon-7b Experiments
 - [x] Create `rand_falcon_dataset5k.csv` where we randomize the `answer ids`, 
       uniformly sampling from the vocabulary (excluding special tokens). 
     - `python3 scripts/randomize_answers.py --input_file results/reachability/k10_falcon7b_wiki5k.csv --output_file results/controllability/rand_falcon_dataset5k.csv --model tiiuae/falcon-7b`
     - `python3 scripts/randomize_answers.py --input_file results/reachability/k10_llama_7b_wiki5k.csv --output_file results/controllability/rand_llama_dataset5k.csv --model huggyllama/llama-7b`
     - `python3 scripts/randomize_answers.py --input_file results/reachability/k10_falcon_40b_wiki5k.csv --output_file results/controllability/rand_falcon40b_dataset5k.csv --model tiiuae/falcon-40b`