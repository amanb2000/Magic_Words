"""
This script accepts a dataframe with the following columns:
 - question
 - question_ids
 - answer
 - answer_ids
 - ...

And adds the following columns: 
 - base_logits      # 1-dim list of the logits of the answer_ids given the question_ids
 - base_rank        # base rank of the answer_ids given the question_ids
"""

import argparse
import os
import sys 
import pdb
from tqdm import tqdm
import math

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from scripts.reachability import load_model, load_input_df

def parse_args(): 
    # get the input_csv path, the model name, output path
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input CSV file. Must have columns `question`, `question_ids`, `answer`, and `answer_ids`.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to output CSV file.')
    parser.add_argument('--model', type=str, required=True,
                        choices=['falcon-7b', 'falcon-40b', 'llama-7b', 'gpt-2-small'], 
                        help='Name of HuggingFace model.')

    parser.add_argument('--rank_only', action='store_true',
                        help='If true, only include the rank (not the full logits) in the output CSV.')
    
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



def get_logits(question_ids, model): 
    """ This function returns the logits of the answer_ids given the question_ids. 

    Returns the logits as a 1-dimensional regular python list of floats.
    """
    # assertions on question_ids handled in `load_input_df`
    # assertions on model handled in `load_model`
    # now we get the logits
    input_ids = torch.tensor(question_ids).unsqueeze(0).to(model.device) # [1, num_toks]
    with torch.no_grad():
        outputs = model(input_ids) # [1, num_toks, vocab_size]
        logits = outputs.logits[0, -1, :] # [vocab_size]
    return logits.tolist()


def log_sum_exp(logits):
    """Compute log-sum-exp in a numerically stable way."""
    max_logit = max(logits)
    sum_exp = sum(math.exp(l - max_logit) for l in logits)
    return max_logit + math.log(sum_exp)

def softmax(logits):
    """Compute softmax values using log-sum-exp for numerical stability."""
    lse = log_sum_exp(logits)
    return [math.exp(l - lse) for l in logits]

def entropy_of_logits(logits):
    """Compute the entropy of a list of logits using softmax."""
    probabilities = softmax(logits)
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return entropy

def get_rank(logits, desired_word): 
    """ logits: 
    """
    if type(desired_word) == list: 
        assert len(desired_word) == 1, "desired_word must be a single integer"
        desired_word = desired_word[0]
    # Check if desired_word is within the valid range
    if not (0 <= desired_word < len(logits)):
        raise ValueError("desired_word is out of range")

    # Get the logit value for the desired_word
    desired_logit = logits[desired_word]

    # Count how many words have a higher logit value
    rank = sum(1 for logit in logits if logit > desired_logit)

    return rank

def add_logits_and_rank(input_df, model, tokenizer, rank_only=False): 
    """ This function adds columns `base_logits` and `base_rank` to the input_df. 
    """
    # assertions on input_df handled in `load_input_df`
    # now we iterate through the rows and add the logits and rank 
    logits_list = []
    rank_list = []
    entropies = []
    num_iters = len(input_df)

    for i, row in tqdm(input_df.iterrows(), total=num_iters):
        # get the question and answer ids
        question_ids = row['question_ids'] # list of ints 
        answer_ids = row['answer_ids'] # int

        # get the logits
        logits = get_logits(question_ids, model)

        # get the rank
        rank = get_rank(logits, answer_ids)

        # get the entropy 
        entropy = entropy_of_logits(logits)

        # append to the lists
        logits_list.append(logits)
        rank_list.append(rank)
        entropies.append(entropy)

    # add the columns to the dataframe
    input_df['base_logits'] = logits_list
    input_df['base_rank'] = rank_list
    input_df['base_entropy'] = entropies

    return input_df


def main(): 
    args = parse_args()

    # get the CSV -- assertions on existing columns are handled here already
    input_df = load_input_df(args.input_file)

    # load the model 
    model, tokenizer = load_model(args.model)

    # add the new columns
    print("Computing the logits and ranks...")
    new_df = add_logits_and_rank(input_df, model, tokenizer)
    print("Done.")

    # output the new dataframe
    print("Outputting CSV...")
    new_df.to_csv(args.output_file, index=False, lineterminator='\n')
    print("Done.")





if __name__ == '__main__':
    main()