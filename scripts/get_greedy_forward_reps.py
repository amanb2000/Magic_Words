"""
get_greedy_forward_reps.py

Given the path to a results folder produced by `scripts/greedy_forward_singe.py`, 
and a desired `y_ids` to extract all equivalent u representations for, this 
stores a jsonl file to disk with the following format for each row:

```json
{
    'y': {y_ids}, 
    'y_str': {tokenizer.decode(y_ids)}
    'u_list': token ids for the given equivalent u that reaches y given x_0, 
    'u_str_list': {tokenizer.batch_decode(u_list)}, 
    ...
    ''
}
```
"""

import argparse
import os
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from tqdm import tqdm 
import gzip


def parse_args(): 
    """
    """
    parser = argparse.ArgumentParser(description='Get the final token reps for all equivalent u for a given y given the output folder of `scripts/greedy_forward_single.py`')
    parser.add_argument('--source_folder', type=str, help='Output folder of `scripts/greedy_forward_single.py')
    parser.add_argument('--y_ids', type=int, help='specific y_ids (int for now) for which we want to compute reps for equivalent u')
    parser.add_argument('--max_eq_u', type=int, help='maximum number of equivalent u to get the reps of.')

    args = parser.parse_args()
    return args

def load_results(results_folder):
    """
    Load the results from the specified folder.
    
    Args:
        results_folder (str): Path to the results folder.
        
    Returns:
        tuple: A tuple containing args, Y_to_U, R_t, and U_t.
    """
    with open(os.path.join(results_folder, "args.json"), "r") as f:
        args = json.load(f)
        
    with open(os.path.join(results_folder, "Y_to_U.json"), "r") as f:
        Y_to_U_ = json.load(f)
        # convert all keys to ints
        Y_to_U = {int(k): v for k, v in Y_to_U_.items()}
        
    with open(os.path.join(results_folder, "R_t.json"), "r") as f:
        R_t = json.load(f)
        
    with open(os.path.join(results_folder, "U_t.json"), "r") as f:
        U_t = json.load(f)

    # check if x_0_ids.json is in the results folder. if its there, 
    # load it. If not, just set x_0_ids to -1
    if os.path.exists(os.path.join(results_folder, "x_0_ids.json")):
        with open(os.path.join(results_folder, "x_0_ids.json"), "r") as f:
            x_0_ids = json.load(f)
    else:
        x_0_ids = -1

        
    return args, Y_to_U, R_t, U_t, x_0_ids

def get_final_token_reps(Y_to_U, x_0_ids, model, tokenizer, max_debug=-1, max_eq_u = 100):
    """
    Get the final token representations for the given Y_to_U (dict[int, List[int]]).
    x_0_ids should be a tensor of shape [1, seq_len] on the model device.

    Returns: 
        final-token_reps: List[Dict[str, Any]] with dict_keys(['y', 'u_list',
        'y_str', 'u_str_list', 'final_token_rep']) where 'final_token_rep' 
        is a list of length num_layers with each element being a list of length hidden_size 
        corresponding to the final token reps at the given layer. 
    """
    final_token_reps = []
    cnt = 0 
    for y, u_list in tqdm(Y_to_U.items()):
        subcnt=0
        for u_spec in tqdm(u_list['all']):
            u_tensor = torch.tensor(u_spec).unsqueeze(0).to(model.device)
            input_ids = torch.cat([u_tensor, x_0_ids], dim=-1)
            attn_mask = input_ids != tokenizer.pad_token_id
            # print("Attention mask: ", attn_mask)
            # print("input_ids: ", input_ids)
            # print("u_tensor: ", u_tensor)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states # tuple of length num_layers. 
                # each outputs.hidden_states[i] is tuple of length batch_size
                # each outputs.hidden_states[i][0] is of shape [seq_len, hidden_size]
                # since we are passing a single input, we only have one element in the tuple 
                #   outputs.hidden_states[i]
                # we want to grab the final token representations for each layer

                final_token_rep = []
                for i in range(len(hidden_states)):
                    final_token_rep.append(hidden_states[i][0][-1, :].cpu().numpy().tolist())

            final_token_reps.append({
                "y": y,
                "u_list": u_spec,
                "y_str": tokenizer.decode(y),
                "u_str_list": [tokenizer.decode(u) for u in u_spec],
                "final_token_rep": final_token_rep, 
                "x_0_str": tokenizer.decode(x_0_ids[0]), 
                "x_0": x_0_ids[0].cpu().numpy().tolist()
            })
            cnt += 1
            if max_debug > 0 and cnt >= max_debug:
                break

            subcnt+=1 
            if subcnt > max_eq_u: 
                break
    return final_token_reps

def get_num_eq_u_per_y(Y_to_U, tokenizer):
    """
    Get the number of equivalent u per y in the given Y_to_U.
    
    Args:
        Y_to_U (dict): Dictionary mapping y to equivalent u.
        
    Returns:
        dict: Dictionary mapping y to the number of equivalent u.
    """
    num_eq_u_per_y = {}
    for y in Y_to_U.keys():
        num_eq_u_per_y[y] = {'num_eq_u': len(Y_to_U[y]['all']), 'y_str': tokenizer.decode(y)}
    num_eq_u_per_y = dict(sorted(num_eq_u_per_y.items()))
    return num_eq_u_per_y



def main(): 
    args = parse_args()
    print(f"Loading results from {args.source_folder}...")
    og_args, Y_to_U, R_t, U_t, x_0_ids = load_results(args.source_folder)
    print(f"Done loading results!\n")

    print("x_0_ids: ", x_0_ids)
    print("og_args['x_0']: ", og_args['x_0'])


    print("Getting number of equivalent u per y...")
    num_eq_u_per_y = get_num_eq_u_per_y(Y_to_U, tokenizer)
    if not(args.y_ids in num_eq_u_per_y.key()): 
        print(f"Desired y_ids {args.y_ids} not found in reachable set.")
        print(f"Please select a y_ids from the following: ", num_eq_u_per_y)
    else: 
        print("y_ids found in reachable set with {num_eq_u_per_y[args.y_ids]} equivalent u")
    print("Done getting number of equivalent u per y!\n")

    
    model_name = og_args['model_name']
    print(f"\nLoading model ", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Done loading model!\n")

    num_u_to_sample = args.max_eq_u
    y_ids = args.y_ids
    all_u = Y_to_U[y_ids]['all']
    if len(all_u) < num_u_to_sample: 
        print("[Warning] num_u_to_sample > number of avaialable u")
        u_sampled = all_u
    else: 
        u_sampled = random.sample(all_u, num_u_to_sample)
    
    Y_to_U_sampled = {y_ids: {
        'first': Y_to_U[y_ids]['first'], 
        'all': u_sampled
    }}

    sampled_u_final_token_reps = get_final_token_reps(Y_to_U_sampled, x_0_ids, model, tokenizer, max_debug=-1, max_eq_u = num_u_to_sample)

    # output sample u final token reps to disk 
    output_path = os.path.join(args.source_folder, f'sampled_u_final_token_reps_top{num_u_to_sample}_yids{y_ids}.jsonl')

    print(f"Saving final token representations to disk at {output_path}")
    with open(output_path, "w") as f:
        # json.dump(sampled_u_final_token_reps, f)
        # write jsonlines 1 at a time 
        for urep in tqdm(sampled_u_final_token_reps): 
            json_line = json.dumps(urep.tolist())  # Convert tensor to list and then to JSON string
            f.write(json_line + '\n')  # Write the JSON string followed by a newline

    print("Done. Have a nice day.\n")

if __name__ == "__main__": 
    main()