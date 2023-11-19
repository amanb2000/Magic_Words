import os
import math
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

def _naive_ingest(model, tokenizer, 
                  reachable_df: pd.DataFrame, 
                  prompt_ids: torch.Tensor, 
                  answer_ids: torch.Tensor, 
                  base_logits: torch.Tensor, 
                  question_ids: torch.Tensor, 
                  question: str): 
    """ Adds rows to `reachable_df` from `prompt_ids` if the corresponding 
    `answer_ids` is novel (i.e., not contained in `reachable_df`). 

    `reachable_df` is a pandas dataframe with columns:
        [question, question_ids, answer, answer_ids, best_prompt,
        best_prompt_ids, base_loss, search_method, prompt_length, prompted_loss,
        base_correct, prompt_correct, question_length]

    `prompt_ids` has shape [batch, prompt_length] (torch.Tensor)
    `answer_ids` has shape [batch, 1] (torch.Tensor)
    `question_ids` has shape [1, question_length] (torch.Tensor)
    """
    assert prompt_ids.shape[0] == answer_ids.shape[0], "prompt_ids and answer_ids must have the same batch dimension."
    assert prompt_ids.shape[1] == 1
    assert question_ids.shape[0] == 1

    batch = prompt_ids.shape[0]

    # R_t is the reachable set. We will keep it updated as we add new rows 
    # to reachable_df.
    R_t = {x for x in reachable_df['answer_ids'].tolist()} 

    for i in tqdm(range(batch)): 
        # Let's check of answer_ids[i] is in R_t. If so, we continue. 
        if answer_ids[i].item() in R_t:
            continue

        print("\tNew answer: ", tokenizer.decode(answer_ids[i].item()))


        # If we're here, we must add a new row to reachable_df.
        _question = tokenizer.batch_decode(question_ids)[0] # str
        assert _question == question, "question_ids and question must match."
        _answer_ids = answer_ids[i].item() # int
        _answer = tokenizer.decode(_answer_ids) # str

        best_prompt_ids = prompt_ids[i,:] # [prompt_length] tensor
        best_prompt = tokenizer.batch_decode(best_prompt_ids)[0] # str

        base_loss = torch.nn.functional.cross_entropy(base_logits[:, -1, :], torch.tensor([_answer_ids]).to(model.device)).item()

        search_method = 'forward'
        prompt_length = best_prompt_ids.shape[0] #CHECK 

        # compute prompted logits
        input_ids = torch.cat([best_prompt_ids.unsqueeze(0), question_ids], dim=-1)
        with torch.no_grad():
            logits = model(input_ids).logits

        prompted_loss = torch.nn.functional.cross_entropy(logits[:, -1, :], torch.tensor([_answer_ids]).to(model.device)).item()
        base_correct = False
        prompt_correct = True
        question_length = question_ids.shape[1]


        new_row = { 'question': question, #str
                    'question_ids': question_ids[0].tolist(), # 1-dim list
                    'answer': _answer,
                    'answer_ids': _answer_ids, # int'
                    'best_prompt': best_prompt, 
                    'best_prompt_ids': best_prompt_ids.tolist(),
                    'base_loss': base_loss, # float
                    'search_method': search_method, # str
                    'prompt_length': prompt_length, # int
                    'prompted_loss': prompted_loss, #float
                    'base_correct': base_correct, # bool, false
                    'prompt_correct': prompt_correct, # bool, true
                    'question_length': question_length
                    }
        # pdb.set_trace()
        print("Comparing new_row's dtypes with reachable df: ")
        new_df = pd.DataFrame([new_row])
        # for key in new_row.keys(): 
        #     print("")
        #     print(f"{key}: {type(new_row[key])} vs. {type(reachable_df[key].iloc[0])}")
        #     print(f"{key}: {type(new_row[key])} vs. {type(reachable_df[key].dtype)}")
        #     print(f"{key}: {new_row[key]} vs. {reachable_df[key].tolist()[0]}")

        # add the new row to reachable_df

        reachable_df_ = pd.concat([reachable_df, new_df], ignore_index=True)
        reachable_df = reachable_df_

        # add the new answer_ids to R_t
        R_t.add(_answer_ids)
    
    return reachable_df






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
                     max_parallel=300):
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
        new_reachable_set = get_reachable_set(x_0, model, tokenizer, 
                                              max_prompt_tokens=max_prompt_tokens, 
                                              max_parallel=max_parallel)

        # add the reachable set to reachable_df
        # df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        reachable_df = pd.concat([reachable_df, new_reachable_set], ignore_index=True)

    return reachable_df