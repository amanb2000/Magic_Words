import torch 
import numpy 
from tqdm import tqdm
import pdb 



def get_batch_logits(model, tokenizer, x_0_ids, alt_prompt_ids, max_parallel=100): 
    """
    Get the logits for a batch of input_ids while not exceeding 
    max_parallel. 

    x_0_ids: [1, num_toks] imposed state sequence 
    alt_prompt_ids: [num_prompts, num_toks] alternative prompt sequences to prepend

    Returns: 
    logits_list: List of logits for each prompt in alt_prompt_ids (only the final token logits)
    """

    logits_list = []

    num_prompts = alt_prompt_ids.shape[0]
    num_batches = ((num_prompts-1) // max_parallel) + 1


    for batch in tqdm(range(num_batches)): 
        start_idx = batch * max_parallel
        end_idx = min((batch+1) * max_parallel, num_prompts)

        alt_prompt_ids_batch = alt_prompt_ids[start_idx:end_idx, :]

        # concatenate with repeated x_0_ids 
        num_in_batch = alt_prompt_ids_batch.shape[0]
        input_ids_batch = torch.cat([alt_prompt_ids_batch,
            x_0_ids.repeat(num_in_batch, 1).to(model.device)], dim=1)

        with torch.no_grad(): 
            attention_mask = (input_ids_batch != tokenizer.pad_token_id).float()
            input_ids_batch = input_ids_batch.to(model.device)
            attention_mask = attention_mask.to(model.device)
            logits = model(input_ids_batch, attention_mask=attention_mask).logits
            logits_list.append(logits[:, -1:, :].cpu())
    # concatenate logits_list on dimension 0 
    logits_list = torch.cat(logits_list, dim=0)
    
    return logits_list

def compute_reachability_loss(logits, R_t, push=1.0, pull=1.0): 
    """
    Computes -CE_loss(logits, unif(R_t)) + H(logits[~R_t])

    logits: [num_prompts, 1, vocab_size=probs]

    Currently only works for single-token answers/reachable sets
    """
    # get the unreached logits 
    reached_logits_unif = [1.0 if i in R_t else 0.0 for i in range(logits.shape[-1])]
    reached_logits_unif = torch.tensor(reached_logits_unif, dtype=torch.float).to(logits.device)
    # normalize 
    reached_logits_unif = reached_logits_unif / reached_logits_unif.sum()
    # add a batch dimension
    reached_logits_unif = reached_logits_unif.unsqueeze(0) # [1, vocab_size]
    # repeat for num_prompts = logits.shape[0]
    reached_logits_unif = reached_logits_unif.repeat(logits.shape[0], 1)

    # compute the cross entropy loss, indexing logits[:, 0, :]
    push_losses = -push*torch.nn.functional.cross_entropy(logits[:, 0, :], reached_logits_unif, reduction='none') # minimize this -- make it very negative -> far from reached set.
    # push_losses has shape [num_prompts]

    # compute the entropy of the logits[~R_t]
    # we also want to compute add a term corresponding to the entropy of 
    # the logits[unreached_logits]. We want this to be a sharply peaked 
    # distribution, meaning we want to minimize its entropy H(). 
    unreached_idx = [i if i not in R_t else -1 for i in range(logits.shape[-1])]
    unreached_idx = [i for i in unreached_idx if i != -1]
    # use this to grab the unreached logits
    unreached_logits = logits[:,0, unreached_idx] # [num_prompts, num_unreached_logits]
    # softmax it 
    unreached_probs = torch.nn.functional.softmax(unreached_logits, dim=-1) # [num_prompts, num_unreached_logits]
    # compute entropy -- want to minimize this
    entropy = -torch.sum(unreached_probs * torch.log(unreached_probs + 1e-8), dim=-1) # [num_prompts]

    pull_losses = pull*entropy


    return push_losses + pull_losses

def batched_compute_reachability_loss(logits, R_t, push=1.0, pull=1.0, max_parallel=5000):
    """
    Computes -CE_loss(logits, unif(R_t)) + H(logits[~R_t]) in batches.
    
    logits: [num_prompts, 1, vocab_size=probs]
    Currently only works for single-token answers/reachable sets
    max_parallel: Maximum number of prompts to process in parallel
    """
    num_prompts = logits.shape[0]
    batch_size = max_parallel
    num_batches = (num_prompts + batch_size - 1) // batch_size
    
    losses = []
    
    with tqdm(total=num_prompts, desc="Computing reachability loss") as pbar:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_prompts)
            
            batch_logits = logits[start_idx:end_idx]
            batch_losses = compute_reachability_loss(batch_logits, R_t, push, pull)
            
            losses.append(batch_losses)
            
            pbar.update(end_idx - start_idx)
    
    losses = torch.cat(losses, dim=0)
    
    return losses




def update_R_U_t(R_t, U_t, Y_to_U, reached, U):
    """
    Updates R_t, U_t based on reached[num_prompt] and 
    U[num_prompt, num_toks]
    """
    cnt=0
    for r in reached.tolist():
        if r not in R_t:
            R_t.add(r)
            U_t.append(U[cnt, :].cpu().tolist())
            Y_to_U[r] = U[cnt, :].cpu().tolist()
        cnt+=1

    return R_t, U_t, Y_to_U

def update_pool_scores(model, tokenizer, x_0_ids, pool, R_t, max_parallel, push=1.0, pull=1.0): 
    """
    Updates the pool scores based on the new reachable set R_t
    """
    pool_scores = []

    # 1: get max prompt length in pool 
    max_prompt_length = max([p.shape[1] for p in pool])
    # 2: pad each prompt in the pool to max_prompt_length
    pool_padded = [] 
    for p in pool: 
        pool_padded.append(torch.cat([torch.ones(1, max_prompt_length - p.shape[1], dtype=torch.long).to(model.device)*tokenizer.pad_token_id, p], dim=1))
    # 3: concatenate all the padded prompts in the pool
    pool_padded = torch.cat(pool_padded, dim=0)

    # 4: get the logits for each prompt in the pool
    logits = get_batch_logits(model, tokenizer, x_0_ids, pool_padded, max_parallel=max_parallel)
    # 5: compute the reachability loss for each prompt in the pool
    reachability_losses = batched_compute_reachability_loss(logits, R_t, push=push, pull=pull)
    return reachability_losses 

@torch.no_grad()
def greedy_forward_reachability(model, tokenizer, x_0, 
                                max_prompt_length=5, 
                                max_iters=100, 
                                max_parallel=100, 
                                pool_size=100, 
                                push=1.0, 
                                pull=1.0, 
                                frac_ext=0.01, 
                                rand_pool=True, 
                                add_special_tokens=False):
    """
    Performs open-ended greedy forward reachability analysis on an LLM.
    
    Args:
        model: The LLM model to analyze (e.g. GPT-2)
        tokenizer: Tokenizer for the LLM
        x_0: The initial state string to start reachability analysis from
        max_prompt_length: The maximum number of tokens to allow in the control prompt 
        max_iters: The maximum number of iterations to run the reachability analysis for
        push: how much weight toward moving away from known reachable set
        pull: how much weight toward moving toward a sharply peaked distribution over the unreachable set
        frac_ext: what fraction of the vocabulary we randomly sample to back-extend the prompt
        rand_pool: do we select a random entry from the pool or the max scoring entry? 
        add_special_tokens: whether to add special tokens to the prompt (e.g., BOS -- defaults to False)
        
    Returns:
        R_t: A set of reachable token ids (ints). 
        U_t: A list of 1-dimensional token ids representing prompts u that each steer to 
            one of the y in R_t.
        Y_to_U: A dictionary mapping each reachable token id y to the prompt u that steers to it.
        x_0_ids: A 1-dim list of token ids representing the initial state sequence.
    """
    # [1, num_toks]
    x_0_ids = tokenizer.encode(x_0, return_tensors='pt', add_special_tokens=add_special_tokens)

    if tokenizer.pad_token_id is None: 
        tokenizer.pad_token_id = tokenizer.eos_token_id

    Y_to_U = {} # dict mapping output values Y to the prompts that get you there
    R_t = set() # set of 
    U_t = [] # List of torch tensors (all on the CPU) -- prompts that get you an R_t member.

    # pool of prompts we are considering/extending back 1 token at a time
    pool_list = []
    pool_scores = []

    # logits has shape [num_prompts, ans_toks=1, vocab_size]
    print("Computing logits for all single-token prompts...")
    # let U be a vocab_size x 1 shaped tensor from 0 ... vocab_size
    # TODO: get rid of //5, this is for debugging only
    U = torch.arange(tokenizer.vocab_size, device=model.device).unsqueeze(1)

    logits = get_batch_logits(model, tokenizer, x_0_ids, U, max_parallel=max_parallel)
    reached = logits[:, 0, :].argmax(-1) # 1-dim tensor
    print("Done!\n")

    print("Updating reachable sets...")
    R_t, U_t, Y_to_U = update_R_U_t(R_t, U_t, Y_to_U, reached, U) 
    print("Done!\n")
    #U_t id a set of lists, R_t is a set of ints
    print("Computing reachability losses for all single-token prompts...")
    reachability_losses = batched_compute_reachability_loss(logits, R_t)
    print("Done!\n")
    # shape [num_prompts]

    # get sorted idx for reachability_losses 
    sort_idx = torch.argsort(reachability_losses)

    # get the top pool_size prompts
    for i in range(min(pool_size, sort_idx.shape[0])): 
        pool_list.append(U[sort_idx[i], :].unsqueeze(0))

    pool_scores = reachability_losses[sort_idx[:pool_size]] # [pool_size]


    for t in range(max_iters):
        # get the top prompt in the pool
        if not rand_pool:
            top_idx = torch.argmin(pool_scores)
        else: 
            top_idx = torch.randint(0, pool_size, (1,))[0]

        top_prompt = pool_list[top_idx] # 
        U_ = torch.arange(tokenizer.vocab_size, device=model.device).unsqueeze(1)

        if frac_ext > 0.0: 
            # randomly sample frac_ext of the vocab to extend the prompt
            # shuffle arange(vocab_size) and use it to index U_
            U_ = U_[torch.randperm(U_.shape[0]), :]
            U_ = U_[:int(frac_ext*U_.shape[0]), :]


        U = torch.concat([U_, top_prompt.repeat(U_.shape[0], 1)], dim=1)

        print(f"\nRound {t}: Top prompt is {tokenizer.batch_decode(top_prompt)[0]}.")
        print("Computing logits for all single-token prompts back-extensions...")
        logits = get_batch_logits(model, tokenizer, x_0_ids, U, max_parallel=max_parallel)
        reached = logits[:, 0, :].argmax(-1)
        print("Done!")

        # add new reached tokens to R_t and U_t
        print("Updating reachable set starting from size ", len(R_t))
        # R_t, U_t = update_R_U_t(R_t, U_t, reached, U) 
        R_t, U_t, Y_to_U = update_R_U_t(R_t, U_t, Y_to_U, reached, U) 
        print("New reachable set size: ", len(R_t))

        print("Computing scores for all new prompts...")
        reachability_losses = batched_compute_reachability_loss(logits, R_t, push=push, pull=pull)
        print("Done!")

        # update pool scores based on new reachable set
        print("Updating pool scores with new R_t...")
        pool_scores = update_pool_scores(model, tokenizer, x_0_ids, pool_list, R_t, max_parallel)
        print("Done!")

        # make new pool with all current pool members and all new U, reachability_losses 
        print("Updating pool...")
        pool_add = []
        for i in range(U.shape[0]): 
            pool_add.append(U[i, :].unsqueeze(0))
        pool_list = pool_list + pool_add
        pool_scores = torch.cat([pool_scores, reachability_losses])

        # resort pool, pool_scores and cut down to pool_size
        sort_idx = torch.argsort(pool_scores)
        pool_list = [pool_list[i] for i in sort_idx[:pool_size]]
        pool_scores = pool_scores[sort_idx[:pool_size]]
        print("Done!")

    return R_t, U_t, Y_to_U, x_0_ids[0, :].cpu().tolist()
