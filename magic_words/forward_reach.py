import os
import argparse 
import pandas as pd
import pdb

from tqdm import tqdm

import torch 
import transformers 
from transformers import AutoTokenizer, AutoModelForCausalLM



def _get_prompt_ids_brute_force(tokenizer): 
    """ Gets a tensor of shape `[vocab_size, 1]` with all token ids 
    for the model. 
    """
    prompt_ids = torch.tensor(range(tokenizer.vocab_size)).unsqueeze(-1)
    return prompt_ids


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