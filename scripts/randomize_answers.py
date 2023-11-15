"""
Call this script on a CSV file with the following columns: 
 - question (str)
 - question_ids (list{int})
 - answer (str) 
 - answer_ids (list{int})

This script will randomize the answer_ids and update the answer column
accordingly. 

Note that we will avoid the `special_tokens` in `tokenizer` when randomizing.
"""

import argparse
import os 
from transformers import AutoTokenizer
import pandas as pd
import torch
import numpy as np
import ast

import pdb

# Argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='Path to input CSV file.')
    parser.add_argument('--output_file', type=str, help='Path to output CSV file.')
    parser.add_argument('--model', type=str, help='Name of HuggingFace model to use for tokenizer.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to use for torch.')
    args = parser.parse_args()
    return args

def get_randomized_df(df, tokenizer): 
    """ This function accepts a dataframe that has (at least) the rows 
    `question`, `question_ids`, `answer`, and `answer_ids` and returns a new
    dataframe with the same rows but with randomized `answer_ids` and `answer`
    columns. 
    
    We also double check that the tokenized version of `question` is actually the 
    same as `question_ids` and that the decoded version of `question_ids` is 
    actually the same as `question`.
    """
    # assert that the columns exist
    assert 'question' in df.columns, "Must have a `question` column!"
    assert 'question_ids' in df.columns, "Must have a `question_ids` column!"
    assert 'answer' in df.columns, "Must have an `answer` column!"
    assert 'answer_ids' in df.columns, "Must have an `answer_ids` column!"

    # assert that the question_ids and answer_ids are lists of ints
    assert type(ast.literal_eval(df['question_ids'][0])) == list, "question_ids must be a list of ints!"
    assert type(df['answer_ids'][0]) == np.int64 or type(df['answer_ids'][0]) == int, "answer_ids must be a list of ints!"
    assert type(ast.literal_eval(df['question_ids'][0])[0]) == int, "question_ids must be a list of ints!"

    # initialize an empty dataframe
    new_df = pd.DataFrame(columns=['question', 'question_ids', 'answer', 'answer_ids'])

    # generate a list of the allowable tokens
    special_tokens = tokenizer.all_special_tokens
    allowable_tokens = [i for i in range(tokenizer.vocab_size) if i not in special_tokens]
    print("Allowable tokens length: ", len(allowable_tokens))


    # loop through the rows of the dataframe
    for i, row in df.iterrows(): 
        # get the question and answer strings
        question = row['question']
        answer = row['answer']
        print("Old answer: ", answer)
        print("Old answer_ids: ", row['answer_ids'])

        # get the question_ids and answer_ids
        question_ids = ast.literal_eval(row['question_ids'])
        answer_ids = int(row['answer_ids']) # int/int64
        # check that the tokenized version of question is the same as question_ids
        tokenized_question = tokenizer.decode(tokenizer.encode(question))
        assert tokenized_question == question, "Tokenized question does not match question_ids!"

        # check that the decoded version of question_ids is the same as question
        decoded_question_ids = tokenizer.decode(torch.tensor(question_ids))
        assert decoded_question_ids == question, "Decoded question_ids does not match question!"

        # check that the tokenized version of answer is the same as answer_ids
        if type(answer) != float: 
            tokenized_answer = tokenizer.decode(tokenizer.encode(answer, return_tensors='pt')[0])
            assert tokenized_answer == answer, "Tokenized answer does not match answer_ids!"

            # check that the decoded version of answer_ids is the same as answer
            decoded_answer_ids = tokenizer.decode(torch.tensor(answer_ids))
            assert decoded_answer_ids == answer, "Decoded answer_ids does not match answer!"

            # get the randomized answer_ids
            randomized_answer_ids = torch.randint(0, len(allowable_tokens), (1,)).item() # int

        # get the randomized answer
        randomized_answer = tokenizer.decode(randomized_answer_ids)
        print("New answer: ", randomized_answer)

        # add the row to the new dataframe
        new_row = pd.DataFrame([{'question': question, 
                                'question_ids': question_ids, 
                                'answer': randomized_answer, 
                                'answer_ids': randomized_answer_ids}])

        if len(new_df) == 0: 
            new_df = new_row 
        else: 
            new_df = pd.concat([new_df, new_row])

    assert len(new_df) == len(df), "New dataframe has different length than old dataframe!"

    return new_df


if __name__ == "__main__":
    # 1: Parse args
    args = parse_args()
    # print out the args
    print("Args:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # 2: Load the CSV file 
    df = pd.read_csv(args.input_file, lineterminator='\n')
    # check that the output file doesn't already exist
    assert not os.path.exists(args.output_file), f"Output file {args.output_file} already exists!"

    # 2: Set the random seed in torch random 
    torch.manual_seed(args.seed)

    # 3: Load the tokenizer
    print(f"Loading tokenizer {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, 
                                              add_eos_token=False,
                                              add_bos_token=False)
    print("Tokenizer special tokens: ", tokenizer.all_special_tokens)
    assert type(tokenizer.all_special_tokens) == list, "Must have a list of special tokens!"
    print("Done loading tokenizer!")


    # 4: Randomize the answer_ids
    new_df = get_randomized_df(df, tokenizer)

    # 5: Save the new dataframe
    new_df.to_csv(args.output_file, index=False)







