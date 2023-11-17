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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .reachability import load_model, load_input_df

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

def add_logits_and_rank(input_df, model, tokenizer): 
    """ This function adds columns `base_logits` and `base_rank` to the input_df. 
    """
    # assertions on input_df handled in `load_input_df`
    # now we iterate through the rows and add the logits and rank 
    logits_list = []
    rank_list = []

    for i, row in input_df.iterrows():
        # get the question and answer ids
        question_ids = row['question_ids']
        answer_ids = row['answer_ids']
        pdb.set_trace()

        # get the logits
        logits = get_logits(question_ids, answer_ids, model)

        # get the rank
        rank = get_rank(logits)

        # append to the lists
        logits_list.append(logits)
        rank_list.append(rank)




def main(): 
    args = parse_args()

    # get the CSV -- assertions on existing columns are handled here already
    input_df = load_input_df(args.input_file)

    # load the model 
    model, tokenizer = load_model(args.model)

    # add the new columns
    new_df = add_logits_and_rank(input_df, model, tokenizer)





if __name__ == '__main__':
    main()