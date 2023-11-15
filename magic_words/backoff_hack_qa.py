import torch 
import numpy as np
from .prompt_hack_qa import greedy_prompt_hack_qa_ids
from .easy_gcg import easy_gcg_qa_ids
from .search_limiters import BruteForce


import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

def backoff_hack_qa_ids(question_ids:torch.Tensor,
                        answer_ids:torch.Tensor, 
                        model, 
                        tokenizer, 
                        search_limiter=None,
                        max_parallel=301, 
                        verbose=True,
                        blacklist=[], 
                        greedy_lengths = [1, 2, 3], 
                        gcg_lengths=[4, 6, 8, 10]):
    """Performs backoff prompt optimization for a question and answer pair 
    as described in the Magic Words paper: https://arxiv.org/abs/2310.04444

     1. Checks base correct argmax condition. 
     2. Greedy search for prompt length 1, 2, 3.
     3. Easy-GCG search for prompt length 4, 6, 8, 10.

    Built for the zero-temperature single-output LLM system control tests (i.e., 
    once answer token). 
    """
    return_dict = {
        'optimal_prompt': None, # list of ids [[]]
        'optimal_prompt_length': -1,  # int
        'prompt_loss': -1.0, # float
        'search_method': None, # str 
        'prompt_correct': None, # bool
        'base_loss': -1.0, # float
    }


    question_ids = question_ids.to(model.device)
    answer_ids = answer_ids.to(model.device)
    # Check that question_ids has shape [1, num_question_toks]
    # Check that answer_ids has shape [1, 1]
    assert question_ids.shape[0] == 1
    assert answer_ids.shape[0] == 1
    assert answer_ids.shape[1] == 1

    # Make the bruteforce search limiter 
    if search_limiter is None:
        search_limiter = BruteForce(tokenizer.vocab_size, blacklist=blacklist)


    # First, check the base case.
    if verbose: 
        print("\nComputing base loss...")

    with torch.no_grad(): 
        q_logits = model(question_ids).logits


    base_answer_logits = q_logits[:, -1, :]
    base_answer_pred = base_answer_logits.argmax().item()
    base_correct = base_answer_pred == answer_ids[0]
    base_loss = torch.nn.functional.cross_entropy(base_answer_logits.cuda(), 
                                                  answer_ids[0]).item()
    if base_correct: 
        if verbose:
            print("Model already gets the correct answer!")
        return {
            'optimal_prompt': [], 
            'optimal_prompt_length': 0, 
            'prompt_loss': -1.0, 
            'search_method': 'base',
            'prompt_correct': True, 
            'base_loss': float(base_loss),
            'base_correct': True
        } 
    elif verbose: 
        print("Model does not get the correct answer with no prompt.")


    if verbose: 
        print("Base loss: ", base_loss)

    return_dict['base_loss'] = float(base_loss)
    return_dict['base_correct'] = False

    # Now we perform greedy search. 
    for i in greedy_lengths:  # 1, 2, 3
        if i == 1: 
            pq = question_ids
        else: 
            pq = torch.cat([prior_best_prompt_ids, question_ids], dim=1)
        
        if verbose:
            print(f"\nGreedy search for prompt length {i} with pq={pq}")
        
        new_prompt, _ = greedy_prompt_hack_qa_ids(pq, answer_ids, 
                                                  1, 
                                                  model, 
                                                  tokenizer, 
                                                  search_limiter, 
                                                  max_parallel=max_parallel)
        ppq = torch.cat([new_prompt, pq], dim=1)

        # check if we have the correct answer
        with torch.no_grad(): 
            ppq_logits = model(ppq).logits
        answer_logits = ppq_logits[:, -1, :]
        answer_pred = answer_logits.argmax()
        new_loss = torch.nn.functional.cross_entropy(answer_logits, answer_ids[0]).item()
        if verbose: 
            print("New loss: ", new_loss)
            print("Predicted answer: ", answer_pred.item())
            print("Desired answer: ", answer_ids.item())

        if answer_pred == answer_ids[0]:
            if verbose:
                print(f"Found correct answer at prompt length {i}!")
            return_dict['optimal_prompt'] = ppq[:, :i].tolist()
            return_dict['optimal_prompt_length'] = i
            return_dict['prompt_loss'] = float(new_loss)
            return_dict['search_method'] = 'greedy'
            return_dict['prompt_correct'] = True
            return return_dict

        # now we set the prior_best_prompt_ids 
        prior_best_prompt_ids = ppq[:, :i]
        print(f"Performing another round with prior_best_prompt_ids = {prior_best_prompt_ids}")



    print("\nMoving on to easy-gcg search...")

    for i in gcg_lengths: 
        if verbose: 
            print(f"\n\nRunning easy-GCG search for prompt length {i}...")
        best_prompt_ids = easy_gcg_qa_ids(question_ids, 
                                          answer_ids, 
                                          i, 
                                          model, 
                                          tokenizer, 
                                          top_k=128, 
                                          batch_size=768, 
                                          num_iters=34,
                                          blacklist=blacklist,
                                          max_parallel=max_parallel)
        if verbose:
            print(f"Best_prompt_ids={best_prompt_ids}")
        ppq = torch.cat([best_prompt_ids, question_ids], dim=1)

        # check if we have the correct answer
        with torch.no_grad(): 
            ppq_logits = model(ppq).logits
        answer_logits = ppq_logits[:, -1, :]
        answer_pred = answer_logits.argmax()
        new_loss = torch.nn.functional.cross_entropy(answer_logits, answer_ids[0]).item()
        if verbose:
            print("New loss: ", new_loss)
            print("Predicted answer: ", answer_pred.item())
            print("Desired answer: ", answer_ids.item())
        if answer_pred == answer_ids[0]:
            if verbose:
                print(f"Found correct answer at prompt length {i}!")
            return_dict['optimal_prompt'] = ppq[:, :i].tolist()
            return_dict['optimal_prompt_length'] = i
            return_dict['prompt_loss'] = float(new_loss)
            return_dict['search_method'] = 'gcg'
            return_dict['prompt_correct'] = True
            return return_dict

        if i != 10: 
            print("Could not find a prompt that gets the correct answer. Trying again with more prompt tokens...")

    print("\n\nCould not find a prompt that gets the correct answer. Returning False.")
    return_dict['optimal_prompt'] = ppq[:, :i].tolist()
    return_dict['optimal_prompt_length'] = i
    return_dict['prompt_loss'] = float(new_loss)
    return_dict['search_method'] = 'gcg'
    return_dict['prompt_correct'] = False 
    return return_dict