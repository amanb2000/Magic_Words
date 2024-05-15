"""
Forward GCG applies the AutoPrompt trick to generate 
control input prompts `u` that maximize the reachable set 
of next tokens `y*` for a given initial input `x_0` where 

    y* = argmax_y P(y | u + x_0)

"""

import os
import math
import pandas as pd
import pdb


from tqdm import tqdm

import torch 
import transformers 
from transformers import AutoTokenizer, AutoModelForCausalLM

from magic_words.get_answer import _get_answer_ids, _batch_get_answer_ids
from magic_words.easy_gcg import get_alt_prompt_ids, get_embedding_weights
from magic_words.utils import _naive_ingest






def get_random_row(x_0, model, tokenizer, base_logits,
                   num_tokens=10):
    """ Generates a random prompt `u` of length `num_tokens` and 
    returns the corresponding row for the reachable set dataframe.

    Args:
        x_0: Initial input to the model. 
        model: HuggingFace model. 
        tokenizer: HuggingFace tokenizer. 
        base_logits: logits for the base input `x_0`. Shape [1, num_tokens, vocab_size]
        num_tokens: Number of tokens in the prompt.
    """
    # ensure that x_0 has shape [1, num_question_tokens]
    assert len(x_0.shape) == 2
    assert x_0.shape[0] == 1

    # generate a random prompt of length `num_tokens` by uniformly 
    # sampling from [0, tokenizer.vocab_size)
    u = torch.randint(low=0, high=tokenizer.vocab_size, size=(1, num_tokens)).to(model.device)
    # ensure u is of shape [1, num_prompt_tokens]
    assert len(u.shape) == 2
    assert u.shape[0] == 1

    question = tokenizer.batch_decode(x_0)[0] # str

    # get the answer_ids for the prompt u + x_0
    answer_ids = _get_answer_ids(u, x_0, model, tokenizer) # [[answer_id]]

    base_loss = torch.nn.functional.cross_entropy(base_logits[:,-1,:], answer_ids[0]).item()

    # compute prompted logits 
    input_ids = torch.cat([u, x_0], dim=1)
    with torch.no_grad(): 
        prompted_logits = model(input_ids).logits
    
    prompted_loss = torch.nn.functional.cross_entropy(prompted_logits[:,-1,:], answer_ids[0]).item()
    new_row = {
        'question': question,
        'question_ids': x_0[0].tolist(),
        'answer': tokenizer.decode(answer_ids.item()),
        'answer_ids': answer_ids.item(),
        'best_prompt': tokenizer.batch_decode(u)[0],
        'best_prompt_ids': u[0].tolist(),
        'base_loss': base_loss,
        'search_method': 'forward_gcg',
        'prompt_length': num_tokens,
        'prompted_loss': prompted_loss,
        'base_correct': False, 
        'prompt_correct': True,
        'question_length': x_0.shape[1]
    }
    return new_row, base_logits, question



def get_prompt_divergence_grads(model, tokenizer, start_prompt, x_0, R_t):
    """ Returns the gradients w.r.t. the one-hot representation of the 
    prompt_ids. Loss function is the cross-entropy loss between the logits and 
    uniform(unreached_logits) where unreached_logits is the complement of R_t

    Args:
        model: HuggingFace model. 
        start_prompt: [1, num_prompt_toks] Starting prompt to mutate. 
        x_0: [1, num_question_toks] Initial input to the model. 
        R_t: Set of reachable tokens.
    """
    assert len(start_prompt.shape) == 2 
    assert start_prompt.shape[0] == 1
    assert len(x_0.shape) == 2
    assert x_0.shape[0] == 1

    model.zero_grad()
    # get the embedding weights 
    embed_weights = get_embedding_weights(model)

    # Get the one-hot prompt_ids, compute the embeddings 
    one_hot_prompt_ids = torch.nn.functional.one_hot(start_prompt, num_classes=embed_weights.shape[0]).to(model.dtype) # [batch, seq_len, vocab]
    one_hot_prompt_ids = one_hot_prompt_ids.to(model.device)
    one_hot_prompt_ids.requires_grad_()
    # one_hot_prompt_ids has shape [1, num_tokens, vocab_size]
    input_embeds = one_hot_prompt_ids @ embed_weights # [seq_len, hidden_size] 


    # get the one-hot future_ids, compute the embeddings
    one_hot_fut_ids = torch.nn.functional.one_hot(x_0, num_classes=embed_weights.shape[0]).to(model.dtype) # [batch, seq_len, vocab]
    one_hot_fut_ids = one_hot_fut_ids.to(model.device)
    future_embeds = one_hot_fut_ids @ embed_weights # [seq_len, hidden_size]
    logits = model(inputs_embeds=torch.cat([input_embeds, future_embeds], dim=1)).logits
    # logits has shape [1, num_tokens+num_future_tokens, vocab_size]

    
    # now we compute the loss on the answer_ids
    answer_logits = logits[:, -1, :] # [1, vocab_size]

    # get the unreached logits 
    reached_logits = [1.0 if i in R_t else 0.0 for i in range(answer_logits.shape[-1])]
    reached_logits = torch.tensor(reached_logits, dtype=torch.float).to(model.device)
    # normalize 
    reached_logits = reached_logits / reached_logits.sum()
    # add a batch dimension
    reached_logits = reached_logits.unsqueeze(0) # [1, vocab_size]

    answer_loss = -torch.nn.functional.cross_entropy(answer_logits, reached_logits) # minimize this -- make it very negative -> far from reached set.

    loss = answer_loss.mean()
    loss.backward()
    grads = one_hot_prompt_ids.grad.clone() # [1, num_prompt_tokens, vocab_size]

    return grads, loss.item()



def get_reachable_gcg_set(x_0, model, tokenizer, 
                          top_k=128,
                          num_prompt_tokens=10, 
                          batch_size=768, 
                          num_iters=34,
                          max_parallel=300, 
                          num_init_prompts = 1, 
                          num_to_mutate = 1): 
    """ Performs forward-GCG to find k reachable tokens from x_0.

    Args: 
        x_0: Initial input to the model. 
        model: HuggingFace model. 
        tokenizer: HuggingFace tokenizer. 

        top_k: Number of top token swaps to explore at each iteration. 
        num_prompt_tokens: Number of tokens in the prompts. 
        batch_size: Batch size of alternate prompts we test each generation (sampled from top k swaps). 
        num_iters: Number of iterations to run GCG for. 
        max_parallel: Maximum number of parallel inference calls to perform.
        num_init_prompts: Number of initial random prompts to use for GCG.
        num_to_mutate: Number of prompts to mutate at each iteration (each gets 
        `batch_size` mutants).
    """
    x_0 = torch.tensor([x_0], dtype=torch.long).to(model.device)
    # x_0 must have shape [1, num_question_tokens]
    assert len(x_0.shape) == 2
    assert x_0.shape[0] == 1
    assert num_prompt_tokens > 0

    

    # get the base logits 
    with torch.no_grad(): 
        base_logits = model(x_0).logits

    # add a random starting row to the reachable set
    print("Populating initial random prompts...")
    reachable_df = -1
    for i in tqdm(range(num_init_prompts)):
        new_row, base_logits, question = get_random_row(x_0, model, tokenizer, base_logits)
        if not (type(reachable_df) == pd.DataFrame): 
            reachable_df = pd.DataFrame([new_row])
        else:
            reachable_df = pd.concat([reachable_df, pd.DataFrame([new_row])], ignore_index=True)
    print("Done. Length of ")


    # initialize the reachable set
    R_t = set(reachable_df['answer_ids'].tolist())
    for iter in range(num_iters): 
        # get the starting prompt from reachable_df we are going to mutate 
        # use random uniform selection -- type list[int]
        alt_prompt_ids_list = []
        for i in range(num_to_mutate):
            start_prompt = reachable_df.sample(1, replace=True, random_state=iter)['best_prompt_ids'].tolist()[0]
            start_prompt = torch.tensor([start_prompt], dtype=torch.long).to(model.device)

            # get the top k token swaps for this prompt
            grads, loss = get_prompt_divergence_grads(model, tokenizer, start_prompt, x_0, R_t)

            # get the top k token swaps for this prompt
            assert grads.shape == (1, start_prompt.shape[1], tokenizer.vocab_size)

            # Now we get the top_k most promising swaps for each token. 
            # X[i,:] holds the k most promising token swaps for token i 
            # of the prompt
            X = (-grads).topk(top_k, dim=-1).indices # [1, num_prompt_tokens, top_k]

            alt_prompt_ids = get_alt_prompt_ids(start_prompt, X, batch_size)
            alt_prompt_ids_list.append(alt_prompt_ids) # [batch, num_prompt_tokens]

        alt_prompt_ids = torch.cat(alt_prompt_ids_list, dim=0) # [num_to_mutate * batch_size, num_prompt_tokens]

        # Now we compute the answer_ids for each of these alternate prompts
        # and add them to the reachable set if they are not already in it.
        print("Computing answer_ids for each alt_prompt_ids...")
        answer_ids = _batch_get_answer_ids(alt_prompt_ids, x_0, model, tokenizer, max_parallel=max_parallel)
        print("Done.\n")

        # add the new rows to the reachable set
        # 3: ingest the answer_ids into reachable_df -- only add if `answer_ids` has 
        # not been seen before. 
        reachable_df = _naive_ingest(model, tokenizer, reachable_df, 
                                    alt_prompt_ids, answer_ids, base_logits, 
                                    x_0, question)
        
        # update R_t
        R_t = {x for x in reachable_df['answer_ids'].tolist()}
        print(f"\n\n=== R_t has {len(R_t)} elements ===\n\n")

