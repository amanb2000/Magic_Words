# Controllability Results 

This directory (`results/controllability/`) is where we will store all the results on controlling language models toward **random** final token generation. 

Experiments will be largely similar to the reachability experiments (in `results/reachability`) performed in September -- except that a random token will be selected for `answer_ids`. 


## Setting up Randomized Datasets
  - `python3 scripts/randomize_answers.py --input_file results/reachability/k10_falcon7b_wiki5k.csv --output_file results/controllability/rand_falcon_dataset5k.csv --model tiiuae/falcon-7b`
  - `python3 scripts/randomize_answers.py --input_file results/reachability/k10_llama_7b_wiki5k.csv --output_file results/controllability/rand_llama_dataset5k.csv --model huggyllama/llama-7b`
  - `python3 scripts/randomize_answers.py --input_file results/reachability/k10_falcon_40b_wiki5k.csv --output_file results/controllability/rand_falcon40b_dataset5k.csv --model tiiuae/falcon-40b`


## Falcon-7b Experiments
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/falcon_7b.sh 0 32  # ran at ~4:30a on Wed Nov 15
CUDA_VISIBLE_DEVICES=1 bash scripts/falcon_7b.sh 1 32
CUDA_VISIBLE_DEVICES=2 bash scripts/falcon_7b.sh 2 32
CUDA_VISIBLE_DEVICES=3 bash scripts/falcon_7b.sh 3 32


CUDA_VISIBLE_DEVICES=0 bash scripts/falcon_7b.sh 4 32  # ran at ~4:30a on Wed Nov 15
CUDA_VISIBLE_DEVICES=1 bash scripts/falcon_7b.sh 5 32
CUDA_VISIBLE_DEVICES=2 bash scripts/falcon_7b.sh 6 32
CUDA_VISIBLE_DEVICES=3 bash scripts/falcon_7b.sh 7 32
```


## Llama-7b Experiments
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/llama_7b.sh 0 32  # ran at ~4:30a on Wed Nov 15
CUDA_VISIBLE_DEVICES=1 bash scripts/llama_7b.sh 1 32
CUDA_VISIBLE_DEVICES=2 bash scripts/llama_7b.sh 2 32
CUDA_VISIBLE_DEVICES=3 bash scripts/llama_7b.sh 3 32

CUDA_VISIBLE_DEVICES=0 bash scripts/llama_7b.sh 4 32  # ran at ~4:30a on Wed Nov 15
CUDA_VISIBLE_DEVICES=1 bash scripts/llama_7b.sh 5 32
CUDA_VISIBLE_DEVICES=2 bash scripts/llama_7b.sh 6 32
CUDA_VISIBLE_DEVICES=3 bash scripts/llama_7b.sh 7 32
```