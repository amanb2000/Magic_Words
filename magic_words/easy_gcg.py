"""
This code implements GCG as implemented in the LLM-Attacks paper 
(https://arxiv.org/abs/2307.15043). 

This allows us to run GCG on large models that require multi-GPU to perform 
inference. 

The goal is to offer an interface identical to `greedy_prompt_hack_qa()` in 
`prompt_hack_qa.py`.
"""

import torch
import transformers
from tqdm import tqdm
import numpy as np
import pdb

from magic_words import batch_compute_score, compute_score
from .search_limiters import SearchLimiter


def easy_gcg_qa(question_str:str,
                answer_str:str, 
                num_tokens: int, 
                model,
                tokenizer,
                top_k,
                batch_size=768,
                num_iters=34,
                max_parallel=1000, 
                blacklist=[]):
    """ Entry point method for `easy_gcg_qa_ids` -- handles tokenization for 
    the user. 

    Returns the optimized `prompt_ids` tensor as a shape [1, num_tokens]
    """

    # First, tokenize the future string
    answer_ids = tokenizer.encode(answer_str, return_tensors="pt").to(model.device)
    # we will add tokens to the beginning of this string as we generate.
    # [1, num_future_tokens] 
    question_ids = tokenizer.encode(question_str, return_tensors="pt").to(model.device)

    return easy_gcg_qa_ids(question_ids, 
                            answer_ids,
                            num_tokens,
                            model,
                            tokenizer,
                            top_k,
                            batch_size=batch_size,
                            num_iters=num_iters,
                            max_parallel=max_parallel,
                            blacklist=blacklist)


def easy_gcg_qa_ids(question_ids:torch.Tensor,
                    answer_ids:torch.Tensor,
                    num_tokens: int, 
                    model,
                    tokenizer,
                    top_k,
                    batch_size=768,
                    num_iters=34,
                    max_parallel=1000, 
                    blacklist=[]):
    """
    Performs GCG to optimize the probability of answer_ids given 
    prompt_ids + question_ids. 

    Returns the optimized prompt as a shape [1, num_tokens] torch.Tensor of 
    ids. 

    args: 
        question_ids: [1, num_question_ids]
        answer_ids: [1, num_answer_ids]
        num_tokens: number of tokens to optimize.
        model: Huggingface causal LLM.
        tokenizer: tokenizer for the model.
        top_k: number of top-k swaps to explore for each token in the prompt.
        batch_size: batch size for exploring promising token swaps.
        num_iters: number of iterations to run GCG for.
        max_parallel: maximum number of tokens to test in parallel.
        blackliset: list[int] of token ids to never apply in the swap. 

    Returns the optimized `prompt_ids` tensor as a shape [1, num_tokens]
    """

    # Check the shape of the incoming tensors
    assert question_ids.shape[0] == 1
    assert answer_ids.shape[0] == 1
    assert len(question_ids.shape) == 2
    assert len(answer_ids.shape) == 2

    # First, we need to randomly instantiate the prompt. 
    # `prompt_ids` is a tensor of shape [1, num_tokens] 
    # that samples from [1, tokenizer.vocab_size], avoiding any 
    # tokens in `blacklist`.
    prompt_ids = instantiate_prompt(model, tokenizer, blacklist, num_tokens)
    print("Initial prompt: ", tokenizer.decode(prompt_ids[0].tolist()))
    # prompt_ids has shape [1, num_tokens]

    # now let's compute the score leveraging the `future_mask` and make sure it 
    # gives the same answer. 
    future_mask = get_future_mask(question_ids, answer_ids, model)
    print(future_mask)

    with torch.no_grad():
        init_loss, _ = compute_score(prompt_ids, 
                                torch.cat([question_ids, answer_ids], dim=1), 
                                model,
                                tokenizer,
                                future_mask=future_mask)
    
    print("Initial loss: ", init_loss.item())


    # Now we run the main GCG loop!
    pbar = tqdm(range(num_iters))
    for iteration in pbar:
        # First we compute the gradients for the one-hot encodings of the 
        # current prompt_ids
        grads, loss = get_prompt_grads(model,
                                       prompt_ids,
                                       question_ids,
                                       answer_ids,
                                       future_mask)
        # grads has shape [1, num_prompt_tokens, vocab_size]

        
        # update the progress bar with this round's loss (3 decimals)
        pbar.set_description("Loss: {:.3f}".format(loss))
        assert grads.shape == (1, num_tokens, tokenizer.vocab_size)

        # set the grads of the blacklist tokens to +inf 
        grads[:, :, blacklist] = float('inf')

        # Now we get the top_k most promising swaps for each token. 
        # X[i,:] holds the k most promising token swaps for token i 
        # of the prompt
        X = (-grads).topk(top_k, dim=-1).indices # [1, num_prompt_tokens, top_k]

        alt_prompt_ids = get_alt_prompt_ids(prompt_ids, X, batch_size)

        # now we need to compute the score for each of these alt_prompt_ids
        # in parallel.
        # alt_prompt_ids has shape [batch_size, num_prompt_tokens]
        alt_prompt_ids = alt_prompt_ids.to(model.device)
        alt_scores = batch_compute_score(alt_prompt_ids, 
                                        torch.cat([question_ids, answer_ids], dim=1), 
                                        model,
                                        tokenizer,
                                        future_mask=future_mask,
                                        max_parallel=max_parallel, 
                                        show_progress=False)


        # now we find the best one. 
        # alt_scores is a numpy array with shape [batch_size]
        best_idx = np.argmin(alt_scores)

        # now we update the prompt_ids
        prompt_ids = alt_prompt_ids[best_idx, :].unsqueeze(0)

    return prompt_ids

def get_alt_prompt_ids(prompt_ids,
                       X, 
                       batch_size):
    """ Get `batch_size` randomly generated alternatives to prompt_ids based on 
    X [1, num_prompt_tokens, top_k]. 

    prompt_ids: [1, num_prompt_tokens]
    X: [1, num_prompt_tokens, top_k]
    batch_size: int

    Returns a tensor of shape [batch_size, num_prompt_tokens]
    """
    num_prompt_tokens = prompt_ids.shape[1]
    alt_ids = torch.zeros((batch_size, num_prompt_tokens), dtype=torch.int64).to(prompt_ids.device)

    for batch_el in range(batch_size): 
        swap_idx = torch.randint(0, num_prompt_tokens, (1,)).item()
        k_idx = torch.randint(0, X.shape[-1], (1,)).item()

        swap_token = X[0, swap_idx, k_idx].item()

        alt_ids[batch_el, :] = prompt_ids
        alt_ids[batch_el, swap_idx] = swap_token

    return alt_ids


def get_prompt_grads(model, 
                     prompt_ids, 
                     question_ids, 
                     answer_ids, 
                     future_mask):
    """ Returns the gradients w.r.t. the one-hot representation of the 
    prompt_ids. 

    prompt_ids: [1, num_tokens]
    
    Returns: grads[1, num_tokens, vocab_size], loss
    """
    model.zero_grad()

    # Getting the embedding weights -- depends on the model type
    if str(type(model)).startswith("<class 'transformers_modules.tiiuae.falcon"):
        embed_weights = model.transformer.word_embeddings.weight # [vocab, hidden_size]
    elif str(type(model)).startswith("<class 'transformers.models.gpt2"):
        embed_weights = model.transformer.wte.weight # [vocab, hidden_size]
    elif str(type(model)).startswith("<class 'transformers.models.llama.modeling_llama.LlamaForCausalLM"): 
        embed_weights = model.model.embed_tokens.weight
    else: 
        # Exception: we don't know how to get the embedding weights for this model
        print(model)
        raise Exception(f"Unknown model type: {type(model)}")

    # get the one-hot prompt_ids, compute the embeddings
    one_hot_prompt_ids = torch.nn.functional.one_hot(prompt_ids, num_classes=embed_weights.shape[0]).to(model.dtype) # [batch, seq_len, vocab]
    one_hot_prompt_ids = one_hot_prompt_ids.to(model.device)
    one_hot_prompt_ids.requires_grad_()
    # one_hot_prompt_ids has shape [1, num_tokens, vocab_size]
    input_embeds = one_hot_prompt_ids @ embed_weights # [seq_len, hidden_size] 


    # get the one-hot future_ids, compute the embeddings
    future_ids = torch.cat([question_ids, answer_ids], dim=1)
    one_hot_fut_ids = torch.nn.functional.one_hot(future_ids, num_classes=embed_weights.shape[0]).to(model.dtype) # [batch, seq_len, vocab]
    one_hot_fut_ids = one_hot_fut_ids.to(model.device)
    future_embeds = one_hot_fut_ids @ embed_weights # [seq_len, hidden_size]
    logits = model(inputs_embeds=torch.cat([input_embeds, future_embeds], dim=1)).logits
    # logits has shape [1, num_tokens+num_future_tokens, vocab_size]

    
    # now we compute the loss on the answer_ids
    logits = logits[:, :-1, :] # getting rid of the last token's next token prediction
    answer_logits = logits[:, -answer_ids.shape[1]:, :] # [1, num_answer_tokens, vocab_size]
    flat_answer_logits = answer_logits.reshape(-1, answer_logits.shape[-1]) # [1*num_answer_tokens, vocab_size]

    answer_loss = torch.nn.functional.cross_entropy(flat_answer_logits, answer_ids.reshape(-1), reduction='none')

    loss = answer_loss.mean()
    loss.backward()
    grads = one_hot_prompt_ids.grad.clone() # [1, num_prompt_tokens, vocab_size]

    return grads, loss.item()


def get_future_mask(question_ids, answer_ids, model):
    """ Returns a future mask of shape [1, num_future_tokens] that 
    is all 1s except for the first `num_question_tokens` tokens, which 
    are 0s. 
    """
    future_mask = torch.ones((1, answer_ids.shape[1]+question_ids.shape[1])).to(model.device)
    future_mask[:, :question_ids.shape[1]] = 0
    return future_mask


def instantiate_prompt(model, tokenizer, blacklist, num_tokens): 
    """ Instantiates a prompt tensor of shape [1, num_tokens] that
    samples from [1, tokenizer.vocab_size], avoiding any tokens in
    `blacklist`.
    """
    allowed_ids = set(range(0, tokenizer.vocab_size)) - set(blacklist)
    allowed_ids = torch.tensor(list(allowed_ids)).to(model.device)

    # _preidx denotes that these are indices used to index into the
    # `allowed_ids` tensor.
    prompt_ids_preidx = torch.randint(1, len(allowed_ids), 
                                (1, num_tokens), 
                                device=model.device)

    # now we need to convert these indices into actual token ids
    prompt_ids = allowed_ids[prompt_ids_preidx] 

    for x in blacklist: 
        assert x not in prompt_ids 

    return prompt_ids

    
