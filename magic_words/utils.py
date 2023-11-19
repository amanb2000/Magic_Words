""" Utility functions for `prompt_landscapes`. 

 1. cat_msg_future():       Combining message and future strings during
                            back-generation. 
 2. ...
"""

import pdb 
import torch 
import numpy as np

import torch 
import pandas as pd

from tqdm import tqdm 

def _naive_ingest(model, tokenizer, 
                  reachable_df: pd.DataFrame, 
                  prompt_ids: torch.Tensor, 
                  answer_ids: torch.Tensor, 
                  base_logits: torch.Tensor, 
                  question_ids: torch.Tensor, 
                  question: str, 
                  eps_e: float = 0.0): 
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
    # assert prompt_ids.shape[1] == 1
    assert question_ids.shape[0] == 1

    batch = prompt_ids.shape[0]

    # R_t is the reachable set. We will keep it updated as we add new rows 
    # to reachable_df.
    R_t = {x for x in reachable_df['answer_ids'].tolist()} 

    for i in tqdm(range(batch)): 
        # Let's check of answer_ids[i] is in R_t. If so, we continue. 
        # generate a random number between 0 and 1
        # if it's less than eps_e, then we continue.
        if answer_ids[i].item() in R_t and np.random.rand() > eps_e:
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

def cat_msg_future(message_ids:torch.Tensor, 
        future_ids:torch.Tensor, 
        insertion_point:int): 
    """ Adds `message_ids` to the `future_ids`. If `forward=True`, then the
    `insertion_point` should be the number of message tokens already
    prepended to `future_ids`. If `forward=False` then the insertion point
    is 0.

    Note that this handles broadcasting the batch dimension of 1 of the
    `future_ids` with the >=1 batch dim of `message_ids`. 

    `message_ids`: [batch, num_msg_toks], the set of message we want to try
        adding to the future string. 
    `future_ids`: [1, num_fut_toks], the set of future tokens (potentially
        including previously prepended message tokens). 
    `future_mask`: [1, num_fut_toks], 1 if we care about CE loss on the
        corresponding entry in `future_ids` and zero else. 
    `insertion_point`: Integer, assumed zero-indexing. I.e., 0 corresponds
        to prepending, etc. 
    """
    assert future_ids.shape[0] == 1

    num_fut_toks = future_ids.shape[1]
    num_msg_toks = message_ids.shape[1]
    batch = message_ids.shape[0]

    assert insertion_point <= num_fut_toks # equality -> add to righthand end. 

    pre_fut = future_ids[:, :insertion_point]
    post_fut = future_ids[:, insertion_point:]
    
    pre_fut_repeated = pre_fut.repeat(batch, 1)
    post_fut_repeated = post_fut.repeat(batch, 1)

    result = torch.cat([pre_fut_repeated, message_ids, post_fut_repeated], 1)

    return result