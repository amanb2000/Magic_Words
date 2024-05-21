""" Code for `batch_compute_score` and `batch_compute_dataset` functions. """

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

@torch.no_grad()
def batch_compute_score_dataset(prompt_ids, 
                                question_ids_list, 
                                answer_ids_list, 
                                model, 
                                tokenizer, 
                                max_parallel=100,
                                show_progress=True): 
    """ Given a list of question_ids and answer_ids, compute the score of 
    a prompt `prompt_ids` on the dataset. 

    Args: 

    prompt_ids: tensor of shape [1, num_prompt_tokens]
    question_ids_list: list of tensors of shape [1, num_question_tokens]
    answer_ids_list: list of tensors of shape [1, num_answer_tokens]
    model: Huggingface causal LLM
    tokenizer: Huggingface tokenizer
    max_parallel: maximum number of parallel predictions to make at once
    show_progress: Whether to show a progress bar.
    """
    num_questions = len(question_ids_list)
    num_answers = len(answer_ids_list)
    assert num_questions == num_answers

    # ensure that each question, answer is a tensor of shape [1, num_tokens]
    max_qa_len=-1
    for i in range(num_questions): 
        assert question_ids_list[i].shape[0] == 1
        assert answer_ids_list[i].shape[0] == 1
        if question_ids_list[i].shape[1] + answer_ids_list[i].shape[1] + prompt_ids.shape[1] > max_qa_len: 
            max_qa_len = question_ids_list[i].shape[1] + answer_ids_list[i].shape[1] + prompt_ids.shape[1]
        assert len(question_ids_list[0].shape) == 2
        assert len(answer_ids_list[0].shape) == 2
    if tokenizer.pad_token_id is None: 
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # create a dataset with `input_ids` = prompt_ids + question_ids + answer_ids 
    # and `labels` = [-100, -100, ..., answer_ids], padding to max length, 
    # including positional codes and attention mask. 

    # Create input IDs and labels
    prompt_ids = prompt_ids.to(model.device)
    input_ids = []
    labels = []
    for i in range(num_questions):
        question_ids = question_ids_list[i].to(model.device)
        answer_ids = answer_ids_list[i].to(model.device)
        
        # Concatenate prompt IDs, question IDs, and answer IDs
        # Shape: [1, num_prompt_tokens + num_question_tokens + num_answer_tokens]
        concat_ids = torch.cat((prompt_ids, question_ids, answer_ids), dim=1)
        
        # Create labels: -100 for prompt and question, actual answer IDs for answer
        # Shape: [1, num_prompt_tokens + num_question_tokens + num_answer_tokens]
        label = torch.full_like(concat_ids, -100)
        label[0, -answer_ids.shape[1]:] = answer_ids

        # Pad the concatenated IDs and labels to max_qa_len
        concat_ids = torch.nn.functional.pad(concat_ids, (0, max_qa_len - concat_ids.shape[1]), value=tokenizer.pad_token_id)
        label = torch.nn.functional.pad(label, (0, max_qa_len - label.shape[1]), value=-100)
        
        input_ids.append(concat_ids)
        labels.append(label)
    
    # concat input_ids into a single tensor on dim=0
    input_ids = torch.cat(input_ids, dim=0) # shape [num_questions, max_qa_len]
    labels = torch.cat(labels, dim=0)

    # Create attention mask: 1 for non-padding tokens, 0 for padding tokens
    # Shape: [num_questions, max_qa_len]
    attention_mask = torch.tensor(input_ids != tokenizer.pad_token_id, dtype=torch.long)

    
    # Compute scores in batches
    scores = []
    num_batches = (num_questions + max_parallel - 1) // max_parallel
    for i in tqdm(range(num_batches), disable=not show_progress):
        start = i * max_parallel
        end = min((i + 1) * max_parallel, num_questions)
        
        batch_input_ids = input_ids[start:end, :].to(model.device)
        batch_attention_mask = attention_mask[start:end, :].to(model.device)
        batch_labels = labels[start:end, :].to(model.device)
        
        # Forward pass
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        
        # Extract loss
        loss = outputs.loss
        
        scores.append(loss.item() * (end-start)/num_questions)
    
    # Compute average score
    avg_score = sum(scores)
    
    return avg_score
