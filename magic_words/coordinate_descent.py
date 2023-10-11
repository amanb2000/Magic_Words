# File where we instantiate the models and call the training loop.

# import packages
import torch
import numpy as np
from tqdm import tqdm

import transformers
from transformers import AutoTokenizer
from magic_words.embedding_highjacking_gcg import LLMEmbeddingHighjacking
import pdb

def gcg_prompt_hack(model, tokenizer, future_str, m, batch_size, k, keep_best, temp, max_iters):

    d_model = model.config.hidden_size
    vocab_size = model.config.vocab_size

    future_ids = tokenizer.encode(future_str, return_tensors="pt").to(model.device)
    num_fut_tokens = future_ids.shape[1]
    future_ids_batched = future_ids.repeat(batch_size, 1)

    # Don't input the last token, since we're trying to predict it
    e_hj = LLMEmbeddingHighjacking(model, m, model.device, future_ids_batched[:, :-1])

    # Optimizer only used to compute gradients (so lr doesn't matter)
    optimizer = torch.optim.Adam([e_hj.weo.embeddings_trainable], lr=0.1)
    # compute number of params in e_hj.weo on next line
    num_params = torch.prod(torch.tensor(e_hj.weo.embeddings_trainable.shape)).item()

    print("Number of parameters: {}".format(num_params))

    # Training loop
    for i in range(max_iters):
        print("Iteration: {}".format(i))

        # Run the model
        outputs = e_hj()

        #beta += 1e-3*i

        #pdb.set_trace()
        # TODO flatten logits and future_ids to deal with batch size > 1
        ce_loss = torch.nn.functional.cross_entropy(outputs.logits[0, -num_fut_tokens:]/temp, future_ids[0, :], reduction="none")

        # Compute regularizer loss
        embedding_vectors = e_hj.embedding_word

        embedding_message = e_hj.weo.embeddings_trainable[0]

        total_loss = ce_loss.mean()


        optimizer.zero_grad()
        total_loss.backward()

        message_grads = e_hj.weo.embeddings_trainable.grad[0]

        X_grads = torch.matmul(message_grads, embedding_vectors.T)
        top_k_subs = torch.topk(-X_grads, k=k, dim=1)

        embedding_message_batched = embedding_message[None, :, :].repeat(batch_size, 1, 1)
        #pdb.set_trace()

        if keep_best:
            start_idx = 1
        else:
            start_idx = 0
        for b in range(start_idx, batch_size):
            i = torch.randint(0, m, (1,))
            j = torch.randint(0, k, (1,))

            # Make replacement
            embedding_message_batched[b, i] = embedding_vectors[top_k_subs.indices[i, j]]

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
        batch_losses = batch_losses.reshape(batch_size, num_fut_tokens)

        ce_loss_per_batch_element = batch_losses.mean(dim=1)

        # find the index of the batch element with the lowest loss
        min_loss_idx = torch.argmin(ce_loss_per_batch_element)

        # Swap first row of embedding_message_batched with the row with the lowest loss
        embedding_message_batched[0] = embedding_message_batched[min_loss_idx]

        e_hj.update_embedding_message(embedding_message_batched)

        #pdb.set_trace()

        # Compute the token closest to each vector in embedding_message
        l2dist = (embedding_vectors[:, None, :] - embedding_message[None, :, :]).norm(dim=-1)
        loss_l2dist = torch.min(l2dist, dim=0).values
        message_ids = torch.argmin(l2dist, dim=0, keepdim=True)

        #print("CE loss: {}".format(ce_loss.mean()))
        print("Total loss: {}".format(total_loss))
                # Check that total loss is a number
        if not torch.isfinite(total_loss):
            pdb.set_trace()

        if i == 0:
            initial_loss = total_loss.item()

        # Concatenate with future_ids
        message_str_ids = torch.cat([message_ids, future_ids], dim=1)

        # Decode
        message = tokenizer.decode(message_str_ids[0])

        print("Message and future str: {}".format(message))
    final_loss = total_loss.item()
    return initial_loss, final_loss, message_str_ids

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

    future_str = "The squirrel and the fox had violent tendencies."

    gcg_prompt_hack(model,
                    tokenizer,
                    future_str,
                    m=10,               # Number of tokens in message
                    batch_size=256,     # Batch size for gradient computation
                    k=32,               # Size of candidate replacement set for each token in message
                    keep_best=False,    # Keeps best message each iteration (often leads to premature convergence)
                    temp=1.0,           # Temperature for ce loss when deciding which token to replace
                    max_iters=1000)     # Number of iterations to run for