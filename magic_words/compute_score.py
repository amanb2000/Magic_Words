"""
This function computes the score of a list of `message` strings 
given a `future` string.
"""
import torch
import pdb

from magic_words import utils

def compute_score(messages_ids:torch.Tensor, 
                  future_ids:torch.Tensor, 
                  model, 
                  tokenizer, 
                  future_mask = None, 
                  insertion_point = 0):
    """Given a message and a future string, compute the log 
    probability of the `future` given the `message`. 

    `messages_ids`: [batch, num_message_tokens]
    `future_ids`: [1, num_future_tokens]

    `insertion_point`: Where do we insert the `message_ids` inside `future_ids`? 
        Defaults to 0 (we prepend the messages). 

    Returns a tuple of (log prob future, log prob all tokens). 
    """
    assert future_ids.shape[0] == 1 

    if future_mask is None: 
        future_mask = torch.ones((1, future_ids.shape[1])).to(model.device)

    assert future_mask.shape == (1, future_ids.shape[1])
    
    num_msg = messages_ids.shape[0] # batch

    messages_ids = messages_ids.to(model.device)
    future_ids = future_ids.to(model.device)


    full_ids = utils.cat_msg_future(messages_ids, 
                                    future_ids,
                                    insertion_point)


    multi_future_ids = full_ids[:, -future_ids.shape[1]:]

    # pdb.set_trace()


    # Now we run the full_ids through the model, grabbing the
    # logits from the output
    #print(full_ids.shape)
    output = model(full_ids, labels=full_ids)
    logits = output.logits

    msg_len = messages_ids.shape[1]

    # computing loss 
    future_logits = logits[:, (msg_len-1):-1, :] # [batch, num_future_tokens, vocab_size]
    batch, num_fut_toks, vocab_size = future_logits.shape

    # multi_future_ids
    flat_future_logits = future_logits.reshape(batch*num_fut_toks, vocab_size)
    
    losses = torch.nn.functional.cross_entropy(flat_future_logits, multi_future_ids.reshape(multi_future_ids.shape[0]*multi_future_ids.shape[1]), reduction='none')
    losses = losses.reshape(batch, num_fut_toks)

    # mask out the losses for the padding tokens
    future_bitmask = future_mask == 1.0 
    future_bitmask = future_bitmask[0,:]
    losses = losses[:, future_bitmask]

    losses = torch.mean(losses, dim=1) # [batch]

    # compute the log probability of the future given the message
    # by summing the log probabilities of each token in the future

    return losses.to('cpu'), output.loss.to('cpu').item() # log prob future, log prob all tokens