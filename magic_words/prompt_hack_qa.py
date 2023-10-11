# Code for iteratively generating optimal prompts to elicit some response 
# from the LLM.

import torch 
from tqdm import tqdm 
import numpy as np
import pdb

from magic_words import batch_compute_score
from .search_limiters import SearchLimiter

def greedy_prompt_hack_qa(question_str:str,
                       answer_str:str, 
                       num_tokens: int, 
                       model,
                       tokenizer,
                       search_limiter:SearchLimiter,
                       max_parallel=1000):
    """ Greedily generate a prompt that maximizes the probability of the
    answer string given the question string. 

        argmax_{prompt} P(answer_str | prompt + question_str)

    Returns a set of tokens (torch.Tensor) that maximizes the probability of
    the future string and a list of anti-logits for each token in the prompt.

    question_str: String between the [prompt] and the [answer] that may 
        affect P(answer_str | prompt + question_str). We DO NOT care about 
        P(question_str | prompt). 
    answer_str: this is what we optimize the probability of. 
    num_tokens: number of greedily generated prompt tokens. 
    model:      Huggingface causal LLM. 
    tokenizer:  tokenizer for the model. 
    search_limiter: SearchLimiter class for determining which tokens are 
        tested during back-generation. 
    """
    # First, tokenize the future string
    answer_ids = tokenizer.encode(answer_str, return_tensors="pt").to(model.device)
    # we will add tokens to the beginning of this string as we generate.
    # [1, num_future_tokens] 
    question_ids = tokenizer.encode(question_str, return_tensors="pt").to(model.device)

    return greedy_prompt_hack_qa_ids(question_ids, 
                                     answer_ids,
                                     num_tokens,
                                     model,
                                     tokenizer,
                                     search_limiter,
                                     max_parallel=max_parallel)

def greedy_prompt_hack_qa_ids(question_ids:torch.Tensor,
                       answer_ids:torch.Tensor, 
                       num_tokens: int, 
                       model,
                       tokenizer,
                       search_limiter:SearchLimiter,
                       max_parallel=1000):

    future_ids = torch.cat([question_ids, answer_ids], axis=1)
    num_answer_ids = answer_ids.shape[1]

    # storing the messages scores from each iteration here.
    message_scores_list = [] 

    # Now, we'll iteratively generate the prompt.
    for i in range(num_tokens): 
        # start by computing the future mask -- we only care about predicting
        # the last `num_fut` tokens. 
        future_mask = torch.zeros(future_ids.shape) 
        future_mask[:, -num_answer_ids:] = 1.0

        # Let's get the candidate messages from our `search_limiter`: 
        message_ids = search_limiter.get_candidates(future_ids, future_mask).to(model.device)

        # Compute the anti-logits for the current future string
        #pdb.set_trace()
        message_scores = batch_compute_score(message_ids,
                                      future_ids, 
                                      model, 
                                      tokenizer, 
                                      future_mask=future_mask,
                                      max_parallel=max_parallel)

        message_scores_list.append(np.expand_dims(message_scores, 0))
        # [vocab_size]

        # Find the token that maximizes the anti-logits
        best_idx = message_scores.argmin()
        best_id = message_ids[best_idx, 0].item()
        print("Best token: ", tokenizer.decode([best_id]))
        print("Token score: ", message_scores.min())

        # Update the future string
        future_ids = torch.cat([torch.tensor([[best_id]]).to(model.device), future_ids], dim=1)
        # [1, num_future_tokens + 1]

    # reverse the message_scores_list 
    # all_message_scores -- now it's in lexicographic order. 
    all_message_scores = np.concatenate(message_scores_list, axis=0)
    return future_ids[:, :num_tokens], all_message_scores
    