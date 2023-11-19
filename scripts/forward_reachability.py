"""
In this script, we will search the space of prompts using Monte Carlo sampling. 
We will save the prompts that yield novel next-token predictions. 

Given a dataset of x_0 (i.e., initial "question_ids"), we search through the 
space of prompts u, trying to maximize the reachable set of y* we get with our 
library of u values. 
"""

import os
import argparse 
import pandas as pd
import pdb

from tqdm import tqdm

import torch 
import transformers 
from transformers import AutoTokenizer, AutoModelForCausalLM

from scripts.reachability import load_model, load_input_df
from scripts.generate_deep_dive import sample_x_0

from magic_words import forward_generate

# from magic_words import _

def parse_args(): 
    # Let's parse the arguments right here: 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input CSV file. Must have columns `question`, `question_ids`, `answer`, and `answer_ids`.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to output CSV file.')
    parser.add_argument('--model', type=str, required=True,
                        choices=['falcon-7b', 'falcon-40b', 'llama-7b', 'gpt-2-small'], 
                        help='Name of HuggingFace model.')

    # Arguments for how we perform our search
    parser.add_argument('--max_prompt_tokens', type=int, default=10,
                        help='Maximum number of tokens allowed in the prompt.')
    parser.add_argument('--max_parallel', type=int, default=300,
                        help='Maximum number of parallel searches to perform. Default=300')

    # Worker numbering
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers to use for skipping rows. Default=1')
    parser.add_argument('--worker_num', type=int, default=0, help='Worker number for skipping rows. Must be in [0, num_workers). Default=0')


    # How do we sample the unique states from the dataframe 
    parser.add_argument('--num_unique_states', type=int, default=250,
                        help='Number of unique states to select from the input dataset.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    
    args = parser.parse_args()

    # ensure that the output_file ends in csv and is in a valid directory and doesn't exist
    assert args.output_file.endswith('.csv'), "Output file must end in .csv"
    assert not os.path.exists(args.output_file), "Output file already exists. Please choose a new file name."
    # make sure the directory exists
    assert os.path.exists(os.path.dirname(args.output_file)), "Output directory does not exist."

    # print out the args nicely
    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    return args







def main(): 
    # parse args
    args = parse_args()

    # Load the input dataframe 
    print(f"\nLoading input dataframe `{args.input_file}`")
    df_in = load_input_df(args.input_file)
    print("Done.")


    # sample the dataframe for `num_unique_states` unique states -- making 
    # sure to evenly sample among each question_length 
    # Length: num_unique_states
    print("\nSampling unique states x_0...")
    unique_states = sample_x_0(df_in, args.num_unique_states, seed=args.seed)
    print("Done.")


    # load model
    print(f"\nLoading model `{args.model}`...")
    model, tokenizer = load_model(args.model)
    print("Done.")


    print(f"\nGenerating reachable set...")
    reachable_df = forward_generate(unique_states, model, tokenizer, 
                                    max_prompt_tokens = args.max_prompt_tokens,
                                    max_parallel = args.max_parallel)
    print("Done.")

    # save the reachable set
    print(f"\nSaving reachable set to `{args.output_file}`...")
    reachable_df.to_csv(args.output_file, index=False)
    print("Done.")





if __name__ == "__main__": 
    main()
