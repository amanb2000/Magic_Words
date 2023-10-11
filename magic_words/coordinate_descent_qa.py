# File where we instantiate the models and call the training loop.

# import packages
import torch
import numpy as np
from tqdm import tqdm

import transformers
from transformers import AutoTokenizer
from magic_words.embedding_highjacking_gcg import LLMEmbeddingHighjacking
import pdb


# All should be working now. But test more extensively before pushing! Important!

def gcg_prompt_hack_qa(model, tokenizer, question_str, answer_str, m, batch_size, batches_before_swap, k, keep_best, temp, max_iters, blacklist):
    question_ids = tokenizer.encode(question_str, return_tensors="pt").to(model.device)
    answer_ids = tokenizer.encode(answer_str, return_tensors="pt").to(model.device)

    return gcg_prompt_hack_qa_ids(model, tokenizer, question_ids, answer_ids, m, batch_size, batches_before_swap, k, keep_best, temp, max_iters, blacklist)


def gcg_prompt_hack_qa_ids(model, tokenizer, question_ids, answer_ids, m, batch_size, batches_before_swap, k, keep_best, temp, max_iters, blacklist): 
    """
    question_ids is of shape [1, num_question_ids]
    answer_ids is of shape [1, num_answer_ids]
    """
    assert question_ids.shape[0] == 1
    assert answer_ids.shape[0] == 1
    assert len(question_ids.shape) == 2
    assert len(answer_ids.shape) == 2


    d_model = model.config.hidden_size
    vocab_size = model.config.vocab_size

    num_question_tokens = question_ids.shape[1]
    num_answer_tokens = answer_ids.shape[1]
    num_fut_tokens = num_question_tokens + num_answer_tokens
    
    future_ids = torch.cat([question_ids, answer_ids], dim=1)
    future_ids_batched = future_ids.repeat(batch_size, 1)

    # Don't input the last token, since we're trying to predict it
    e_hj = LLMEmbeddingHighjacking(model, m, model.device, future_ids_batched[:, :-1])

    # Optimizer only used to compute gradients (so lr doesn't matter)
    optimizer = torch.optim.Adam([e_hj.weo.embeddings_trainable], lr=0.1)
    # compute number of params in e_hj.weo on next line
    num_params = torch.prod(torch.tensor(e_hj.weo.embeddings_trainable.shape)).item()

    #print("Number of parameters: {}".format(num_params))

    top_k_subs = torch.zeros((m, k), dtype=torch.long, device=model.device)

    # Training loop
    most_promising_swap = torch.zeros((1, m), dtype=torch.long, device=model.device)
    most_promising_loss = 1e10
    best_message = torch.zeros((1, m), dtype=torch.long, device=model.device)
    best_loss = 1e10
    embedding_vectors = e_hj.embedding_word
    for i in tqdm(range(max_iters)):
        #print("Iteration: {}".format(i))

        # Compute regularizer loss

        embedding_message = e_hj.weo.embeddings_trainable[0]

        embedding_message_batched = embedding_message[None, :, :].repeat(batch_size, 1, 1)

        # Compute swaps with tokens in candidate set
        if i > 0:
            if keep_best:
                start_idx = 1
            else:
                start_idx = 0
            for b in range(start_idx, batch_size):
                m_rand = torch.randint(0, m, (1,))
                k_rand = torch.randint(0, k, (1,))

                # Make replacement
                embedding_message_batched[b, m_rand] = embedding_vectors[top_k_subs.indices[m_rand, k_rand]]

        # Recompute loss
        e_hj.update_embedding_message(embedding_message_batched)
        #pdb.set_trace()
        outputs = e_hj()

        # Must first flatten for batch size > 1
            # computing loss 
        future_logits = outputs.logits[:, -num_fut_tokens:, :] # [batch, num_future_tokens, vocab_size]

        # multi_future_ids
        flat_future_logits = future_logits.reshape(batch_size*num_fut_tokens, vocab_size)
        
        batch_losses = torch.nn.functional.cross_entropy(flat_future_logits, future_ids_batched.reshape(future_ids_batched.shape[0]*future_ids_batched.shape[1]), reduction='none')
        batch_losses = batch_losses.reshape(batch_size, num_fut_tokens)[:, -num_answer_tokens:]

        ce_loss_per_batch_element = batch_losses.mean(dim=1)

        # find the index of the batch element with the lowest loss
        min_loss_idx = torch.argmin(ce_loss_per_batch_element)
        min_loss = ce_loss_per_batch_element[min_loss_idx]

        # Replace most promising swap if min loss lower
        if min_loss < most_promising_loss:
            #print("setting most promising swap")
            most_promising_loss = min_loss
            most_promising_swap = embedding_message_batched[min_loss_idx]
        if min_loss < best_loss:
            best_loss = min_loss
            best_message = embedding_message_batched[min_loss_idx]

        if i % batches_before_swap == 0:
            # Swap in most promising swap
            embedding_message_batched[0] = most_promising_swap
            e_hj.update_embedding_message(embedding_message_batched)

            # Recompute loss
            outputs = e_hj()

            ce_loss = torch.nn.functional.cross_entropy(outputs.logits[0, -num_fut_tokens:]/temp, future_ids[0, :], reduction="none")

            total_loss = ce_loss[-num_answer_tokens:].mean()
            #print("Iteration: {}, Total loss: {}".format(i, total_loss))

            optimizer.zero_grad()
            total_loss.backward()

            message_grads = e_hj.weo.embeddings_trainable.grad[0]

            X_grads = torch.matmul(message_grads, embedding_vectors.T)

            # Create mask for blacklist
            blacklist_mask = torch.zeros((m, vocab_size), dtype=torch.bool, device=model.device)
            blacklist_mask[:, blacklist] = True
            # Set to max float value for the datatype
            X_grads[blacklist_mask] = torch.finfo(X_grads.dtype).max

            top_k_subs = torch.topk(-X_grads, k=k, dim=1)

            most_promising_loss = 1e10
        else:
            embedding_message_batched[0] = embedding_message

        # Compute the token closest to each vector in embedding_message
        l2dist = (embedding_vectors[:, None, :] - embedding_message_batched[0][None, :, :]).norm(dim=-1)
        message_ids = torch.argmin(l2dist, dim=0, keepdim=True)

        #print("CE loss: {}".format(ce_loss.mean()))
        if i % batches_before_swap == 0:
            print("Total loss: {}".format(total_loss))

        if i == 0:
            initial_loss = total_loss.item()

        # Concatenate with future_ids
        message_str_ids = torch.cat([message_ids, future_ids], dim=1)

        # Decode
        message = tokenizer.decode(message_str_ids[0])
        # if i % batches_before_swap == 0:
            # print("Message and future str: {}".format(message))
            # print("Message ids: ", message_ids)

    
    final_loss = best_loss.item()
    l2dist = (embedding_vectors[:, None, :] - best_message[None, :, :]).norm(dim=-1)
    message_ids = torch.argmin(l2dist, dim=0, keepdim=True)
    best_message_str_ids = torch.cat([message_ids, future_ids], dim=1)

    return initial_loss, final_loss, best_message_str_ids

if __name__ == "__main__":
    # Create dataloader

    # Initialize a tokenizer and model
    model_name = "tiiuae/falcon-7b"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model = pipeline.model
    model.eval()

    question_str = "The squirrel and the fox had violent tendencies."
    answer_str = "Oh"

    initial_loss, final_loss, best_message = gcg_prompt_hack_qa(model,
                    tokenizer,
                    question_str,
                    answer_str,
                    m=8,                    # Number of tokens in message
                    batch_size=96,         # Batch size for gradient computation
                    batches_before_swap=8,  # Number of batches before swapping in new message
                    k=128,                   # Size of candidate replacement set for each token in message
                    keep_best=False,        # Keeps best message each iteration (often leads to premature convergence)
                    temp=1.0,               # Temperature for ce loss when deciding which token to replace
                    max_iters=400,
                    blacklist=[2, 3, 4])         # Number of iterations to run for

    print("Initial loss: {}".format(initial_loss))
    print("Final loss: {}".format(final_loss))
    print("Best message: ", tokenizer.decode(best_message[0]))