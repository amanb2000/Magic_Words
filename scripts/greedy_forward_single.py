"""
This script performs open-ended greedy forward reachability analysis on an LLM.
"""

import argparse
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from magic_words import greedy_forward_reachability

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True,
                        help='Name or path of the LLM model to analyze (e.g., meta-llama/Meta-Llama-3-8B).')
    parser.add_argument('--x_0', type=str, required=True,
                        help='The initial state string to start reachability analysis from.')
    parser.add_argument('--max_prompt_length', type=int, default=5,
                        help='The maximum number of tokens to allow in the control prompt. Default: 5')
    parser.add_argument('--max_iters', type=int, default=100,
                        help='The maximum number of iterations to run the reachability analysis for. Default: 100')
    parser.add_argument('--max_parallel', type=int, default=100,
                        help='The maximum number of parallel runs to do. Default: 100')
    parser.add_argument('--pool_size', type=int, default=100,
                        help='The size of the pool for selecting prompts. Default: 100')
    parser.add_argument('--push', type=float, default=1.0,
                        help='Weight for moving away from known reachable set. Default: 1.0')
    parser.add_argument('--pull', type=float, default=1.0,
                        help='Weight for moving toward a sharply peaked distribution over the unreachable set. Default: 1.0')
    parser.add_argument('--frac_ext', type=float, default=0.01,
                        help='Fraction of the vocabulary to randomly sample to back-extend the prompt. Default: 0.01')
    parser.add_argument('--rand_pool', action='store_true',
                        help='Whether to select a random entry from the pool or the max scoring entry. Default: False')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to store the output files (args.json, R_t, U_t, historical data).')
    parser.add_argument('--add-special-tokens', action='store_true',
                        help='Whether to add special tokens to the prompt. Default: False')

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # save args dictionary to args.json in the output_dir 
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Print out the args nicely
    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    return args

def main():
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    model.half()
    print("Model dtype: ", model.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    R_t, U_t, Y_to_U, x_0_ids = greedy_forward_reachability(
        model, tokenizer, args.x_0,
        max_prompt_length=args.max_prompt_length,
        max_iters=args.max_iters,
        max_parallel=args.max_parallel,
        pool_size=args.pool_size,
        push=args.push,
        pull=args.pull,
        frac_ext=args.frac_ext,
        rand_pool=args.rand_pool,
        add_special_tokens=args.add_special_tokens
    )

    # Save the arguments, R_t, U_t, and historical data to the output directory
    # save Y_to_U dictionary as json 
    print("Saving Y_to_U dictionary as json...")
    with open(os.path.join(args.output_dir, 'Y_to_U.json'), 'w') as f:
        json.dump(Y_to_U, f, indent=4)
    print("Y_to_U dictionary saved as json.")
    
    # save R_t and U_t as json
    R_t = list(R_t)

    print("Saving R_t and U_t as json...")
    with open(os.path.join(args.output_dir, 'R_t.json'), 'w') as f:
        json.dump(R_t, f, indent=4)
    with open(os.path.join(args.output_dir, 'U_t.json'), 'w') as f:
        json.dump(U_t, f, indent=4)
    print("R_t and U_t saved as json.")

    print("Saving x_0_ids as json.")
    with open(os.path.join(args.output_dir, 'x_0_ids.json'), 'w') as f:
        json.dump(x_0_ids, f, indent=4)
    print("x_0_ids saved as json.")

if __name__ == "__main__":
    main()
