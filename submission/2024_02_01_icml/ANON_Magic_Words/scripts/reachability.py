""" This script measures the reachability of some desired next token y given 
some imposed state tokens x and some control tokens u. It attempts to find tokens 
u satisfying 

    y = \arg\max_{y'} P(y' | u + x)

given some dataset of (x, y) pairs.
"""

import argparse
import pandas as pd
import os
import pdb

import torch 
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from magic_words import backoff_hack_qa_ids

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

    parser.add_argument('--max_parallel', type=int, default=301,
                        help='Maximum number of parallel runs to do. Default=301')

    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers to use for skipping rows. Default=1')
    parser.add_argument('--worker_num', type=int, default=0, help='Worker number for skipping rows. Must be in [0, num_workers). Default=0')

    parser.add_argument('--blacklist', type=list, default=[], help='List of tokens to blacklist from the search. Default=[]')
    parser.add_argument('--greedy_lengths', nargs='+', type=int, default=[1,2,3],
                        help='List of lengths to use for greedy search. Default=[1,2,3]')
    parser.add_argument('--gcg_lengths', nargs='+', type=int, default=[4,6,8,10],
                        help='List of lengths to use for GCG search. Default=[4,6,8,10]')

    
    parser.add_argument('--verbose', action='store_true', help='Print out verbose messages. Default=False')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode. Default=False')
    
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

def load_input_df(df_path): 
    # Load the input dataframe, validate it has the right columns
    input_df = pd.read_csv(df_path, lineterminator='\n')
    assert 'question' in input_df.columns, "Input CSV must have column `question`"
    assert 'question_ids' in input_df.columns, "Input CSV must have column `question_ids`"
    assert 'answer' in input_df.columns, "Input CSV must have column `answer`"
    assert 'answer_ids' in input_df.columns, "Input CSV must have column `answer_ids`"

    # ensure type of question is str
    input_df['question'] = input_df['question'].astype(str)
    # ensure type of question_ids is list
    input_df['question_ids'] = input_df['question_ids'].apply(lambda x: eval(x))
    # ensure type of answer is str
    input_df['answer'] = input_df['answer'].astype(str)
    # ensure type of answer_ids is int
    input_df['answer_ids'] = input_df['answer_ids'].astype(int)

    return input_df


def load_model(model_name): 
    if model_name == 'falcon-7b':
        model_name = "tiiuae/falcon-7b"
        print(f"Loading model `{model_name}`...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        model = pipeline.model
        model.eval()
        print("Done loading model and tokenizer!\n")
    elif model_name == 'falcon-40b':
        model_name = "tiiuae/falcon-40b"
        print(f"Loading model `{model_name}`...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        model = pipeline.model
        model.eval()
        print("Done loading model and tokenizer!\n")
    elif model_name == 'llama-7b': 
        model_name = "huggyllama/llama-7b"
        print(f"Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              add_bos_token=False,
                                              add_eos_token=False)
        tokenizer.bos_token = ''
        tokenizer.eos_token = ''
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        model = model.half() # convert to fp16 for fast inference.
        model.eval()
        print("Done loading model and tokenizer!\n")
    elif model_name == "gpt-2-small": 
        model_name = "gpt2"
        print(f"Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # set the pad token as the eos token for the tokenizer 
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to('cuda')
        model = model.half()
        model.eval()
    else: 
        # exception: model not found
        raise ValueError(f"Model `{args.model}` not found. Please choose from `falcon-7b`, `falcon-40b`, `llama-7b`, or `gpt-2-small`.")

    return model, tokenizer


def compute_reachability(input_df, model, tokenizer, blacklist, 
                         greedy_lengths, gcg_lengths, max_parallel=301, 
                         verbose=False, debug=False): 
    # Now we compute reachability for each row in the input dataframe. 
    # We will store the results in a list of dicts, which we will convert to a 
    # dataframe at the end. 
    results_df = None # we will store the results csv dataframe here
    for i, row in input_df.iterrows(): 
        if i % args.num_workers != args.worker_num: 
            continue
        print("\n\n=== Running row {} of {} ===".format(i, len(input_df)))

        if verbose: 
            print(f"\n\nRow {i}: {row['question']}")
            print(f"Answer: {row['answer']}")
            print(f"Answer ids: {row['answer_ids']}")

        # First we run the base model to get the base loss. 
        # We will also check if the base model gets the correct answer. 
        # If it does, we can stop right here. 
        
        # row['question_ids'] is a list of ints.
        # row['answer_ids'] is an int.
        assert type(row['question_ids']) == list
        assert type(row['answer_ids']) == int

        question_ids = torch.tensor(row['question_ids'], dtype=torch.int64).unsqueeze(0) 
        answer_ids = torch.tensor([row['answer_ids']], dtype=torch.int64).unsqueeze(0)

        assert question_ids.shape == (1, len(row['question_ids']))
        assert answer_ids.shape == (1, 1)

        # ensure ids are on the same device as the model 
        question_ids = question_ids.to(model.device)
        answer_ids = answer_ids.to(model.device)


        # Running the backoff_hack_qa_ids function! 
        return_dict = backoff_hack_qa_ids(question_ids, answer_ids, model, tokenizer, 
                                          max_parallel=max_parallel,
                                          verbose=verbose, 
                                          blacklist=blacklist, 
                                          greedy_lengths=greedy_lengths, 
                                          gcg_lengths=gcg_lengths)
        if verbose: 
            print(f"Return dict: {return_dict}")
        
        # Now add to the results dataframe
        results_df = add_row(results_df, row, return_dict)

        if debug: 
            break

    return results_df

def add_row(results_df, row, return_dict): 
    if len(return_dict['optimal_prompt']) > 0 and type(return_dict['optimal_prompt'][0]) == list: 
        return_dict['optimal_prompt'] = return_dict['optimal_prompt'][0]

    if len(return_dict['optimal_prompt']) > 0: 
        best_prompt = tokenizer.batch_decode(return_dict['optimal_prompt'])[0]
    else: # no prompt -- base correct must be true.
        best_prompt = ''

    new_row = pd.DataFrame([{
        'question': row['question'],
        'question_ids': row['question_ids'],
        'answer': row['answer'],
        'answer_ids': row['answer_ids'],


        'base_loss': return_dict['base_loss'],
        'search_method': return_dict['search_method'],
        'best_prompt': best_prompt,
        'best_prompt_ids': return_dict['optimal_prompt'],
        'prompt_length': return_dict['optimal_prompt_length'],
        'prompted_loss': return_dict['prompt_loss'], 
        'base_correct': return_dict['base_correct'],
        'prompt_correct': return_dict['prompt_correct'],
        'question_length': len(row['question_ids'])
    }])

    if results_df is None:
        results_df = new_row
    else: 
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        
    return results_df




if __name__ == '__main__': 
    args = parse_args()

    input_df = load_input_df(args.input_file)

    model,tokenizer = load_model(args.model)

    solved_df = compute_reachability(input_df, model, tokenizer, args.blacklist,
                                        args.greedy_lengths, args.gcg_lengths,
                                        args.max_parallel, verbose=args.verbose, 
                                        debug = args.debug)
    
    print("\n\nSaving dataframe...")
    solved_df.to_csv(args.output_file, index=False)