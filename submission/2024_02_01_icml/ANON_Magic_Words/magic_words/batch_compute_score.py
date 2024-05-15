""" Code for `batch_compute_score` function. """

import torch 
from tqdm import tqdm
import numpy as np
import pdb

from .compute_score import compute_score

@torch.no_grad()
def batch_compute_score(message_ids:torch.Tensor,
                        future_ids:torch.Tensor, 
                        model, 
                        tokenizer, 
                        future_mask = None,
                        max_parallel=100, 
                        insertion_point=0, 
                        show_progress=True):
    """ Computes the score `CE(future_ids | message_ids[i])` for all i in 
    batches of size `max_parallel`. 

    message_ids: [num_message_tokens, 1] message tensor. 
    future_ids: [1, num_future_tokens] input tensor. 
    future_mask: [1, num_future_tokens] mask tensor or None. 1 if we care about 
        predicting that token, 0 if we don't. 
    max_parallel: maximum number of parallel predictions to make at once 
        (depends on GPU/memory/task, usually 100-1000 is good).
    insertion_point: Where do we insert the `message_ids` inside `future_ids`? 
        Defaults to 0 (we prepend the messages). 
    show_progress: Whether to show a progress bar.

    Returns: 1-dim numpy array of shape [vocab_size] containing the anti-logits.
    """
    assert future_ids.shape[0] == 1

    vocab_size = tokenizer.vocab_size
    num_fut = future_ids.shape[1]

    # Generating a `future_mask` if it doesn't exist, else checking it's valid.
    if future_mask is None: 
        future_mask = torch.ones((1, num_fut)).to(model.device)
    else: 
        future_mask = future_mask.to(model.device)
        assert future_mask.shape == (1, num_fut)


    # Setting up for batch-wise loop.
    num_ids = message_ids.shape[0]
    num_iters = num_ids // max_parallel + 1
    mean_loss = 0.0
    future_losses = [] # losses on `future_ids[future_mask]` associated with 
                       # each message_id

    for i in tqdm(range(num_iters), disable=not show_progress):
        start = i * max_parallel 
        end_ = min((i+1) * max_parallel, num_ids)

        msg_scores, avg_loss = compute_score(message_ids[start:end_, :], 
                                        future_ids, 
                                        model, 
                                        tokenizer, 
                                        future_mask=future_mask, 
                                        insertion_point=insertion_point) 

        future_losses.append(msg_scores.to(torch.float32).numpy())


        mean_loss += ((end_-start)/num_iters) * avg_loss

    all_future_losses = np.concatenate(future_losses, axis=0)

    return all_future_losses