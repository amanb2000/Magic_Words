import os
import math
import pandas as pd
import pdb


from tqdm import tqdm

import torch 
import transformers 

def _get_answer_ids(prompt_ids, question_ids, model, tokenizer): 
    """ Runs `[prompt_ids + question_ids]` through the model to get logits, then 
    takes the argmax over the vocab_size dimension to return the set of 
    `answer_ids` as a [batch, 1] tensor. 

    `prompt_ids` has shape [batch, prompt_length]
    `question_ids` has shape [batch, question_length]
    """
    if question_ids.shape[0] == 1 and prompt_ids.shape[0] > 1:
        question_ids = question_ids.repeat(prompt_ids.shape[0], 1)

    assert prompt_ids.shape[0] == question_ids.shape[0], 'Batch dimension of prompt_ids and question_ids must match in _get_answer_ids()'

    # Compute the logits 
    input_ids = torch.cat([prompt_ids, question_ids], dim=-1).to(model.device)
    with torch.no_grad():
        logits = model(input_ids).logits
    
    # Compute the answer_ids
    answer_ids = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)

    # ensure answer_ids has the right dimensions 
    assert answer_ids.shape == (prompt_ids.shape[0], 1), f"answer_ids has shape {answer_ids.shape} but should have shape {(prompt_ids.shape[0], 1)}"

    return answer_ids

def _batch_get_answer_ids(prompt_ids, question_ids, model, tokenizer, max_parallel=300): 
    """ Calls `_get_answer_ids()` in batches of size `max_parallel`. 
    """
    batch = prompt_ids.shape[0]

    # get the number of batches we need 
    num_batches = math.ceil(batch / max_parallel)

    # initialize the answer_ids tensor 
    answer_ids = torch.zeros(batch, 1, dtype=torch.long).to(model.device)

    # loop through each batch
    for i in tqdm(range(num_batches)): 
        # get the batch of prompt_ids and question_ids
        batch_start = i*max_parallel
        batch_end = min((i+1)*max_parallel, batch)

        prompt_ids_batch = prompt_ids[batch_start:batch_end]

        # get the answer_ids for this batch
        answer_ids_batch = _get_answer_ids(prompt_ids_batch, question_ids, model, tokenizer)

        # add the answer_ids_batch to answer_ids
        answer_ids[batch_start:batch_end] = answer_ids_batch

    return answer_ids