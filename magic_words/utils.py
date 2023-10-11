""" Utility functions for `prompt_landscapes`. 

 1. cat_msg_future():       Combining message and future strings during
                            back-generation. 
 2. ...
"""

import pdb 
import torch 
import numpy as np


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