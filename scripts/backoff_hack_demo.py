""" Backoff prompt hack for gauging controllability as in
https://arxiv.org/abs/2310.04444. 

Given a question and answer pair, we want to generate a prompt that will force 
the answer to be the argmax over P(answer | prompt + question). Note that the 
answer is assumed to be 1 token in length.

First we will check the base case: is the answer already the argmax? If so, 
we will return an empty prompt. 

Then, we will see if we can solve it with **greedy prompt search** (see appendix
B of https://arxiv.org/abs/2310.04444), starting with 1 token, then 2, then 3.
For each length, we will check if we have met the argmax condition. If we reach
the argmax condition, we will return the prompt. 

Then, we will perform Greedy Coordinate Gradient search for prompt length 4, 6,
8, and 10 (also in Appendix B of https://arxiv.org/abs/2310.04444).  We will
continue checking at each length for the argmax condition, and return. 
"""
import pdb
import argparse

import torch 
import numpy as np
from magic_words import backoff_hack_qa_ids


import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__": 
    """ Demonstration of the backoff script -- same 
    """
    # Parse the args
    parser = argparse.ArgumentParser()
    # Add the argument
    parser.add_argument('--model', choices=['falcon-7b', 'falcon-40b', 'llama-7b', 'gpt-2-small'],
                        help='The model to use (falcon-7b, falcon-40b, llama-7b, or gpt-2-small)', 
                        default='falcon-7b')
    #seed argument -- int, default to 42
    parser.add_argument('--seed', type=int, default=42, help='The random seed to use with torch (default: 42). Used in GCG algorithm sampling.')
    args = parser.parse_args()

    # set the pytorch seed
    torch.manual_seed(args.seed)


    # get model and tokenizer -- tiiuae/falcon-7b
    if args.model == 'falcon-7b':
        model_name = "tiiuae/falcon-7b"
        print(f"Loading model `{model_name}`...")
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
        print("Done loading model and tokenizer!\n")
    elif args.model == 'falcon-40b':
        model_name = "tiiuae/falcon-40b"
        print(f"Loading model `{model_name}`...")
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
        print("Done loading model and tokenizer!\n")
    elif args.model == 'llama-7b': 
        model_name = "huggyllama/llama-7b"
        print(f"Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              add_bos_token=False,
                                              add_eos_token=False)
        tokenizer.bos_token = ''
        tokenizer.eos_token = ''
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        model = model.half() # convert to fp16 for fast inference.
        model.eval()
        print("Done loading model and tokenizer!\n")
    elif args.model == "gpt-2-small": 
        model_name = "gpt2"
        print(f"Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # set the pad token as the eos token for the tokenizer 
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to('cuda')
        model = model.half()
        model.eval()
    else: 
        # exception: model not found
        raise ValueError(f"Model `{args.model}` not found. Please choose from `falcon-7b`, `falcon-40b`, `llama-7b`, or `gpt-2-small`.")
    




    

    # Get the question-answer pair
    question = "What is the meaning of life? "
    answer = "42"

    print("\nQUESTION: ", question)
    print("ANSWER: ", answer, "\n")

    question_ids = tokenizer.encode(question, return_tensors="pt").to(model.device)
    answer_ids = tokenizer.encode(answer, return_tensors="pt").to(model.device)

    if not (answer_ids.shape[0] == answer_ids.shape[1] == 1): # must be only 1 answer token!
        print(f"[WARNING] Answer {answer} does not correspond to a single token (encoded = {answer_ids})")
        print(f"[WARNING] Cutting off answer_ids at the first token.")
        answer_ids = answer_ids[:, 1:2]
        answer = tokenizer.decode(answer_ids[0].tolist())
        print("[WARNING] New answer: ", answer, "\tAnswer ids: ", answer_ids)



    # question_ids = torch.tensor([[204, 23, 1684, 25, 204, 28245, 56647, 64619]], dtype=torch.int64)
    # answer_ids = torch.tensor([[62469]], dtype=torch.int64)

    print("Question ids: ", question_ids)
    print("Answer ids: ", answer_ids)

    # Call backoff hack on the question-answer pair
    return_dict = backoff_hack_qa_ids(question_ids, answer_ids, model, tokenizer)
    print("Return dictionary: ", return_dict)

    optimal_prompt_str = tokenizer.batch_decode(return_dict['optimal_prompt'])[0]

    print("\n\nDecoded Optimal prompt (u): ", optimal_prompt_str)
    print("Optimal prompt length (tokens, |u|): ", return_dict['optimal_prompt_length'])
    print("Prompt loss: ", return_dict['prompt_loss'])
    if return_dict['prompt_correct']: 
        print("Prompt is correct!")
        print(f"\nTHEREFORE: `{answer}` = argmax_a P(a | `{optimal_prompt_str}` + `{question}`)")
    else: 
        print("Unable to find an optimal prompt that gets the correct answer.\nConsider increasing the maximum allowable prompt length :)")
        print("\nBest prompt found: ", optimal_prompt_str)





 
