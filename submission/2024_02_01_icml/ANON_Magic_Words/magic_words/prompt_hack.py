# Code for iteratively generating optimal prompts to elicit some response 
# from the LLM.

import torch 
from tqdm import tqdm 
import numpy as np
import pdb

from magic_words import batch_compute_score
from magic_words import utils
from .search_limiters import SearchLimiter

def greedy_prompt_hack(future_str:str, 
                       num_tokens: int, 
                       model,
                       tokenizer,
                       search_limiter:SearchLimiter,
                       max_parallel=1000, 
                       forward_gen=False):
    """ Greedily back-generate a prompt that maximizes the probability of the
    future string.

        argmax_{prompt} P(future_str | prompt) 

    Returns a set of tokens (torch.Tensor) that maximizes the probability of
    the future string and a list of anti-logits for each token in the prompt.

    future_str:         String we are trying to optimize a prompt to elicit. 
    num_tokens:         Number of prompt tokens we are generating to elicit 
                        `future_str`. 
    model:              Model we are working with. 
    tokenizer:          Tokenizer for the model. 
    search_limiter:     SearchLimiter object for dictating which subset of 
                        tokens are used in the prompt. 
    max_parallel:       Number of parallel inference calls we perform when 
                        testing out messages. 
    forward_gen:        False for back-generating (pre-pending) each greedily 
                        searched token. True for forward generating the prompt. 

    """
    # First, tokenize the future string
    future_ids = tokenizer.encode(future_str, return_tensors="pt").to(model.device)
    # we will add tokens to the beginning of this string as we generate.
    # [1, num_future_tokens] 

    num_fut = future_ids.shape[1]

    # storing the messages scores from each iteration here.
    message_scores_list = [] 

    # Now, we'll iteratively generate the prompt.
    for i in range(num_tokens): 
        # Computing the insertion point for new tokens based on the value of 
        # `i` and `forward_gen`. 
        if forward_gen: 
            insertion_point = i
        else: 
            insertion_point = 0

        # start by computing the future mask -- we only care about predicting
        # the last `num_fut` tokens. 
        future_mask = torch.zeros(future_ids.shape) 
        future_mask[:, -num_fut:] = 1.0

        # Let's get the candidate messages from our `search_limiter`: 
        message_ids = search_limiter.get_candidates(future_ids, future_mask).to(model.device)

        # Compute the anti-logits for the current future string
        message_scores = batch_compute_score(message_ids,
                                      future_ids, 
                                      model, 
                                      tokenizer, 
                                      future_mask=future_mask,
                                      max_parallel=max_parallel, 
                                      insertion_point=insertion_point)

        message_scores_list.append(np.expand_dims(message_scores, 0))
        # [vocab_size]

        # Find the token that maximizes the anti-logits
        best_id = message_scores.argmin()
        print("Best token: ", tokenizer.decode([best_id]))
        print("Token score: ", message_scores.min())

        # Update the future string
        future_ids_old = torch.cat([torch.tensor([[best_id]]).to(model.device), future_ids], dim=1)
        future_ids = utils.cat_msg_future(torch.tensor([[best_id]]).to(model.device), 
            future_ids,
            insertion_point)

        print("Future_ids: ", future_ids)
        print("Decoded future_ids: ", tokenizer.batch_decode(future_ids))
        # [1, num_future_tokens + 1]

    # reverse the message_scores_list 
    # all_message_scores -- now it's in lexicographic order. 
    all_message_scores = np.concatenate(message_scores_list, axis=0)
    return future_ids[:, :num_tokens], all_message_scores