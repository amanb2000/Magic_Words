#!/bin/bash

# This is the experiment script for Falcon-7b controllability 
# on the randomized wikitext dataset. 
# Run with `falcon_7b [worker num] [num workers]` 
# e.g. `falcon_7b 0 1` for worker 0 of 1

BASE_PATH="results/multi_token/"


INPUT_CSV="mt_falcon_dataset_768.csv"
OUTPUT_CSV="falcon-7b/mt768_${1}_of_${2}.csv"

echo "Input CSV: $BASE_PATH$INPUT_CSV"
echo "Output CSV: $BASE_PATH$OUTPUT_CSV"

MODEL_NAME="falcon-7b"

source /home/paperspace/Magic_Words/venv/bin/activate
source ~/.bash_profile
echo "IFTTT Key: $IFTTT_KEY"

python3 scripts/reachability.py \
    --input_file $BASE_PATH$INPUT_CSV \
    --output_file $BASE_PATH$OUTPUT_CSV \
    --model $MODEL_NAME \
    --greedy_lengths 1 2 3 \
    --gcg_lengths 4 8 16 32 64 \
    --worker_num $1 \
    --verbose \
    --num_workers $2 

curl -X POST -H "Content-Type: application/json" -d "{\"Worker\": \"$(cat ~/WHOAMI)\", \"Name\":\"$OUTPUT_CSV\"}" https://maker.ifttt.com/trigger/finished_experiment/json/with/key/$IFTTT_KEY

git pull
git add $BASE_PATH$OUTPUT_CSV
git commit -m "[$(cat ~/WHOAMI)] Add $BASE_PATH$OUTPUT_CSV"
git push
