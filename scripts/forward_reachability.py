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


def get_reachable_set(x_0, model, tokenizer, max_prompt_tokens=10):
    """ Given a single state x_0, generate the reachable set of y* values by 
    enumerating all possible prompts u of length less than max_prompt_tokens. 
    """
    assert max_prompt_tokens == 1, "`get_reachable_set()` Not implemented with max_prompt_tokens != 1"

    reachable_df = pd.DataFrame(columns=['question', #
                                         'question_ids', #
                                         'answer', #
                                         'answer_ids', #
                                         'base_loss', #
                                         'search_method', #
                                         'prompt_length', # 
                                         'prompted_loss', #
                                         'base_correct', #
                                         'prompt_correct', 
                                         'question_length'])

    # get the base logits 
    x_0 = torch.tensor(x_0).unsqueeze(0).to(model.device)
    with torch.no_grad():
        base_logits = model(x_0).logits # [1, seq_len, vocab_size]

    base_answer_ids = base_logits[0,-1,:].argmax().item()
    base_answer = tokenizer.decode(base_answer_ids) # str

    question = tokenizer.batch_decode(x_0)[0] # 
    for i in tqdm(range(tokenizer.vocab_size)): 
        prompt_ids = torch.tensor([[i]]).to(model.device)
        answer = tokenizer.batch_decode(prompt_ids)[0] # str

        input_ids = torch.cat([prompt_ids, x_0], dim=-1)

        # get the logits out of the model 
        with torch.no_grad():
            logits = model(input_ids).logits

        answer_logits = logits[:, -1, :] # [1, vocab_size] tensor
        answer_ids = answer_logits.argmax().item() # int
        answer = tokenizer.decode(answer_ids) # str

        # if this answer_ids is new, add it to reachable_df. If not, we continue. 
        if answer_ids in reachable_df['answer_ids'].tolist():
            continue
        else: 
            print(f"\t[{i}] Found u that yields new y* = ", answer)
        

        # Compute the base loss on `answer_ids` with `bbase_logits`
        base_loss = torch.nn.functional.cross_entropy(base_logits[:, -1, :], torch.tensor([answer_ids]).to(model.device)).item()

        # Compute the prompted loss on `answer_ids` with `logits`
        prompted_loss = torch.nn.functional.cross_entropy(logits[:, -1, :], torch.tensor([answer_ids]).to(model.device)).item()

        # check if base correct is true 
        base_correct = base_answer_ids == answer_ids

        prompt_correct=True # true by construction

        question_length = x_0.shape[1] # [1, num_toks]

        new_row =  {'question': question, #str
                    'question_ids': x_0[0].tolist(), # 1-dim list
                    'answer': answer, # str
                    'answer_ids': answer_ids, # int
                    'base_loss': base_loss, # float
                    'search_method': 'forward', # str
                    'prompt_length': 1, # int
                    'prompted_loss': prompted_loss, #float
                    'base_correct': base_correct, # bool, false
                    'prompt_correct': prompt_correct, # bool, true
                    'question_length': question_length}

        if len(reachable_df) == 0: 
            reachable_df = pd.DataFrame([new_row])
        else: 
            reachable_df = pd.concat([reachable_df, pd.DataFrame([new_row])], ignore_index=True)

    return reachable_df





        




def forward_generate(unique_states, model, tokenizer, max_prompt_tokens=10):
    """ Given a list of unique states x_0, generate the reachable set of y* values 
    for each x_0. 
    Args:
        unique_states (pd.Dataframe): List of unique states (i.e., x_0 values)
            to generate the reachable set for. Must have a `question_ids` column.
        model (transformers.PreTrainedModel): HuggingFace model.
        tokenizer (transformers.PreTrainedTokenizer): HuggingFace tokenizer.
        max_prompt_tokens (int): Maximum number of tokens allowed in the prompt.

    Returns:
        reachable_df (pd.DataFrame): Dataframe of reachable set. 
    """
    # initialize reachable_df
    reachable_df = pd.DataFrame(columns=['question', 
                                         'question_ids', 
                                         'answer',
                                         'answer_ids',
                                         'base_loss', 
                                         'search_method', 
                                         'prompt_length', 
                                         'prompted_loss', 
                                         'base_correct', 
                                         'prompt_correct', 
                                         'question_length'])

    # loop through each unique state
    # for i, x_0 in enumerate(unique_states['question_ids'].tolist()): 
    for i, row in unique_states.iterrows():
        x_0 = row['question_ids']

        assert type(x_0) == list, "question_ids must be a list of token ids -- not a string."

        print(f"{i+1}/{len(unique_states)}: {row['question']}")

        # get the reachable set for this x_0
        new_reachable_set = get_reachable_set(x_0, model, tokenizer, max_prompt_tokens)

        # add the reachable set to reachable_df
        # df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        reachable_df = pd.concat([reachable_df, new_reachable_set], ignore_index=True)

    return reachable_df




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
    reachable_df = forward_generate(unique_states, model, tokenizer, args.max_prompt_tokens)
    print("Done.")

    # save the reachable set
    print(f"\nSaving reachable set to `{args.output_file}`...")
    reachable_df.to_csv(args.output_file, index=False)
    print("Done.")





if __name__ == "__main__": 
    main()
