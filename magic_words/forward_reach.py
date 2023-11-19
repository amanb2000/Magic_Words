import os
import math
import pandas as pd
import pdb


from tqdm import tqdm

import torch 
import transformers 
from transformers import AutoTokenizer, AutoModelForCausalLM

from magic_words.forward_gcg import get_reachable_gcg_set
from magic_words.get_answer import _get_answer_ids, _batch_get_answer_ids
from magic_words.utils import _naive_ingest


def _get_prompt_ids_brute_force(tokenizer): 
    """ Gets a tensor of shape `[vocab_size, 1]` with all token ids 
    for the model. 
    """
    prompt_ids = torch.tensor(range(tokenizer.vocab_size)).unsqueeze(-1)
    return prompt_ids







def get_base_row(x_0, model, tokenizer): 
    """ Gets the base row for the reachable set. 
    I.e., the row corresponding to null control input `u`.
    """
    # get the base logits 
    x_0 = torch.tensor(x_0).unsqueeze(0).to(model.device)
    with torch.no_grad():
        base_logits = model(x_0).logits # [1, seq_len, vocab_size]

    base_answer_ids = base_logits[0,-1,:].argmax().item()
    base_answer = tokenizer.decode(base_answer_ids) # str
    base_loss = torch.nn.functional.cross_entropy(base_logits[:, -1, :], torch.tensor([base_answer_ids]).to(model.device)).item() 

    question = tokenizer.batch_decode(x_0)[0] # 

    # let's add the first row to the dataframe 
    new_row =  {'question': question, #str
                'question_ids': x_0[0].tolist(), # 1-dim list
                'answer': base_answer, # str
                'answer_ids': base_answer_ids, # int
                'best_prompt': '', 
                'best_prompt_ids': [-1], 
                'base_loss': base_loss, # float
                'search_method': 'forward', # str
                'prompt_length': 0, # int
                'prompted_loss': 0.0, #float
                'base_correct': True, # bool, false
                'prompt_correct': True, # bool, true
                'question_length': x_0.shape[1]}
    return new_row, base_logits, question


def get_reachable_set(x_0, model, tokenizer, max_prompt_tokens=10, max_parallel=300):
    """ Given a single state x_0, generate the reachable set of y* values by 
    enumerating all possible prompts u of length less than max_prompt_tokens. 
    """
    assert max_prompt_tokens == 1, "`get_reachable_set()` Not implemented with max_prompt_tokens != 1"

    reachable_df = pd.DataFrame(columns=['question', #
                                         'question_ids', #
                                         'answer', #
                                         'answer_ids', #
                                         'best_prompt', 
                                         'best_prompt_ids', 
                                         'base_loss', #
                                         'search_method', #
                                         'prompt_length', # 
                                         'prompted_loss', #
                                         'base_correct', #
                                         'prompt_correct', 
                                         'question_length'])

    new_row, base_logits, question = get_base_row(x_0, model, tokenizer) 

    reachable_df = pd.concat([reachable_df, pd.DataFrame([new_row])], ignore_index=True)

    # 1: get the prompt_ids 
    # prompt_ids will have shape [vocab_size, 1].
    prompt_ids = _get_prompt_ids_brute_force(tokenizer).to(model.device)
    x_0 = x_0.to(model.device)

    # 2: get the answer_ids for each prompt_ids
    # answer_ids will have shape [batch, 1]. 
    print("Computing answer_ids for each prompt_ids...")
    answer_ids = _batch_get_answer_ids(prompt_ids, x_0, model, tokenizer, max_parallel=max_parallel)
    print("Done.\n")


    # 3: ingest the answer_ids into reachable_df -- only add if `answer_ids` has 
    # not been seen before. 
    reachable_df = _naive_ingest(model, tokenizer, reachable_df, 
                                 prompt_ids, answer_ids, base_logits, 
                                 x_0, question)
    return reachable_df



def forward_generate(unique_states, model, tokenizer, 
                     max_prompt_tokens=10, 
                     max_parallel=300, 
                     gcg=False, 
                     top_k=128,
                     batch_size=768,
                     num_iters=34):
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
                                         'best_prompt',
                                         'best_prompt_ids',
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
        if gcg: 
            new_reachable_set = get_reachable_gcg_set(x_0, model, tokenizer, 
                                                      top_k=top_k,
                                                      num_prompt_tokens=max_prompt_tokens, 
                                                      batch_size=batch_size, 
                                                      num_iters=num_iters,
                                                      max_parallel=max_parallel)
        else:
            new_reachable_set = get_reachable_set(x_0, model, tokenizer, 
                                                max_prompt_tokens=max_prompt_tokens, 
                                                max_parallel=max_parallel)

        # add the reachable set to reachable_df
        # df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        reachable_df = pd.concat([reachable_df, new_reachable_set], ignore_index=True)

    return reachable_df