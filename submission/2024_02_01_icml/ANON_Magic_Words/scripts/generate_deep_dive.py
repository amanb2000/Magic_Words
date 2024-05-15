"""
# Generate deep dive dataset

This script generates a "deep dive" on a subset of the starting states x_0 
(i.e., the question column) of a reachability dataset. The deep dive generates 
many instance for each unique starting state x_0, where the desired y* value are
evenly spaced along the ranked next-token probability distribution for x_0. 

In short: 
 1. Select some `num_unique_states` from the original dataset. 
 2. Compute the output logits for each x_0 in the selected states. 
 3. Rank the logits to get a ranked list from most likely P(y|x_0) to leaset likely. 
 4. Generate a new dataset with y* values evenly spaced along the ranked list 
    based on some `skip` value. 
"""

import argparse 
import os
import numpy as np
import pandas as pd
import pdb

from scripts.reachability import load_model, load_input_df
from scripts.add_logits import add_logits_and_rank

# import torch 
# from transformers import AutoTokenizer, AutoModelForCausalLM 

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input CSV file. Must have columns `question`, `question_ids`, `answer`, and `answer_ids`.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to output CSV file.')
    parser.add_argument('--model', type=str, required=True,
                        choices=['falcon-7b', 'falcon-40b', 'llama-7b', 'gpt-2-small'], 
                        help='Name of HuggingFace model.')

    # num_unique_states 
    parser.add_argument('--num_unique_states', type=int, default=250,
                        help='Number of unique states to select from the input dataset.')
    parser.add_argument('--skip', type=int, default=265,
                        help='Number of tokens to skip between each y* value.')
    parser.add_argument('--num_shallow', type=int, default=-1,
                        help='Number of shallow dive tokens to select as y* for each x_0. If -1, then use the skip value.')
                    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')

    args = parser.parse_args()

    assert (args.skip>0) != (args.num_shallow>0), "Must specify either skip or num_shallow > 0"

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


def sample_x_0(df_in, num_unique_states, seed=42): 
    """ Samples `num_unique_states` from `df_in` (i.e., rows) making sure that 
    the samples are equally balanced between each of the `question_length` 
    values in the dataset. 
    Args:
    df_in (pd.DataFrame): Input dataframe.
    num_unique_states (int): Number of rows to sample.
    seed (int, optional): Random seed for reproducibility. Defaults to 42.
    
    Returns:
    pd.DataFrame: Sampled dataframe with balanced `question_length` values.
    """
    # Set random seed
    np.random.seed(seed)

    # Calculate the number of samples per question_length group
    num_groups = df_in['question_length'].nunique()
    samples_per_group = num_unique_states // num_groups

    # Sample from each group
    sampled_dfs = []
    for _, group_df in df_in.groupby('question_length'):
        sampled_dfs.append(group_df.sample(n=samples_per_group, replace=False))

    # Concatenate the sampled dataframes
    result_df = pd.concat(sampled_dfs, ignore_index=True)

    return result_df


def _get_deep_row(rank_to_ids, question_ids, tokenizer, skip): 
    """Retrieves the deep dive rows for a single question by sampling ever 
    `skip` tokens from the ranked list of logits
    """
    deep_dive_rows = []
    for i in range(0, len(rank_to_ids), skip): 
        # get the token id 
        answer_id = rank_to_ids[i]

        # get the token
        answer = tokenizer.decode(answer_id)

        # now we copy row, add the token_id and token as answer_ids and answer
        deep_dive_rows.append({
            'question_ids': question_ids, 
            'question': tokenizer.decode(question_ids), 
            'answer_ids': answer_id, 
            'answer': answer, 
            'question_length': len(question_ids), 
            '_base_rank': i
        })
    return deep_dive_rows

def _get_shallow_row(rank_to_ids, question_ids, tokenizer, num_shallow): 
    """ Retrieves the num_shallow most likely next tokens as the y* values for 
    a single question_ids (x_0) token sequence. 
    
    rank_to_ids[rank] = token_id that has rank `rank` in the next token 
    probabilities.
    """
    shallow_dive_rows = []
    for i in range(num_shallow): 
        # get the token id 
        answer_id = rank_to_ids[i]

        # get the token
        answer = tokenizer.decode(answer_id)

        # now we copy row, add the token_id and token as answer_ids and answer
        shallow_dive_rows.append({
            'question_ids': question_ids, 
            'question': tokenizer.decode(question_ids), 
            'answer_ids': answer_id, 
            'answer': answer, 
            'question_length': len(question_ids), 
            '_base_rank': i
        })

    
    return shallow_dive_rows

    

def get_deep_dive(unique_states, tokenizer, skip, num_shallow=-1): 
    """
    Args:
    unique_states (pd.DataFrame): Dataframe with the columns `question_ids`, 
        `base_logits`. 
    skip (int): Number of tokens to skip between each y* value drawn from 
        the ranked list of logits.
    """

    # iterate through the rows
    deep_dive_rows = []

    for _, row in unique_states.iterrows():
        # get the question ids and logits
        question_ids = row['question_ids'] # 1-dim list of ints
        base_logits = row['base_logits'] # 1-dim list of floats, length vocab_size

        assert type(question_ids) == list, "Question ids must be a list."
        assert type(question_ids[0]) == int, "Question ids must be a list of ints."

        assert type(base_logits) == list, "Base logits must be a list."
        assert type(base_logits[0]) == float, "Base logits must be a list of floats."

        # rank_to_ids is a list where rank_to_ids[rank] = token_id that 
        # has rank `rank` in the sorted list of logits. So the most likely (highest logit) 
        # token is rank 0, and the least likely (lowest logit) token is rank vocab_size-1.
        rank_to_ids = np.argsort(base_logits)[::-1].tolist()

        # Now we sample every `skip` tokens from the ranked list of logits.
        if skip > 0: 
            deep_dive_rows += _get_deep_row(rank_to_ids, question_ids, tokenizer, skip)
        elif num_shallow > 0: 
            deep_dive_rows += _get_shallow_row(rank_to_ids, question_ids, tokenizer, num_shallow)
        else: 
            # error -- invalid shallow num_shallow combination 
            raise ValueError(f"Invalid combination of shallow and num_shallow: {shallow}, {num_shallow}")

            

    # convert to dataframe
    deep_dive = pd.DataFrame(deep_dive_rows)
    return deep_dive


def main(): 
    args = parse_args()

    # load the dataset
    print("Loading df...")
    df_in = load_input_df(args.input_file)
    print("Done.")

    # sample the dataframe for `num_unique_states` unique states -- making 
    # sure to evenly sample among each question_length 
    # Length: num_unique_states
    print("\nSampling unique states x_0...")
    unique_states = sample_x_0(df_in, args.num_unique_states, seed=args.seed)
    print("Done.")

    # load the model and tokenizer 
    print("\nLoading model so we can compute logits...")
    model, tokenizer = load_model(args.model)
    print("Done.")

    # Get logits. This will add a column `base_logits` to `unique_states`, which 
    # we will use next to generate the deep dive dataset.
    print("\nComputing logits and ranks of next tokens for each unique state x_0...")
    unique_states = add_logits_and_rank(unique_states, model, tokenizer)
    print("Done.")

    # Now we generate the deep dive dataset.
    # Length: num_unique_states * args.skip
    print("\nGenerating the set of y* for each x_0 using logits = P(y|x_0)...")
    deep_dive = get_deep_dive(unique_states, tokenizer, args.skip, 
                              num_shallow=args.num_shallow)
    print("Done.")

    # save the deep dive dataset
    print("\nSaving the deep dive dataset...")
    deep_dive.to_csv(args.output_file, index=False, lineterminator='\n')
    print("Done.")








if __name__ == "__main__": 
    main()