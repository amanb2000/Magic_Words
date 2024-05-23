"""
sgcg.py 

Script to optimizer a prompt of length `k` for a dataset of question-answer 
pairs using stochastic GCG (SGCG) algorithm. 

Args: 
 - dataset: path to JSON file containing question-answer pairs (default: datasets/squad_train_v2.0.jsonl).
    Must have keys 'question', 'answer', 'context' at least. 
 - out_dir: path to directory to save optimized prompts + logs. 
 - model: huggingface model name (default: meta-llama/Meta-Llama-3-8B-Instruct)
 - k: length of prompt to optimize.

 - top_k: gcg param for number of swaps to consider (default: 128)
 - batch_size: gcg param for batch size for num active prompts (default: 768)
 - num_iters: gcg param for num stochastic local search iterations (default: 100)
 - grad_batch_size: number of training examples to use for each gradient update (default: 4)
"""
import argparse 
import jsonlines
import json
import os
import pdb
import datetime
from tqdm import tqdm 

from transformers import AutoTokenizer, AutoModelForCausalLM

import magic_words

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize a prompt of length k for a dataset of question-answer pairs using SGCG algorithm.")
    parser.add_argument("--dataset", default="datasets/squad_train_v2.0.jsonl", help="Path to JSON file containing question-answer pairs.")
    parser.add_argument("--out_dir", default="None", help="Path to directory to save optimized prompts and logs.")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace model name.")
    parser.add_argument('--max_parallel', default=100, type=int, help="Maximum number of parallel predictions to make at once (default=100).")
    parser.add_argument("--k", type=int, default=20, 
                        help="Length of prompt to optimize (default=20).")
    parser.add_argument("--top_k", type=int, default=128, 
                        help="GCG parameter for number of swaps to consider (default=128).")
    parser.add_argument("--batch_size", type=int, default=768, 
                        help="GCG parameter for batch size for number of active prompts (default=768).")
    parser.add_argument("--num_iters", type=int, default=100, 
                        help="GCG parameter for number of stochastic local search iterations (default=100).")
    parser.add_argument("--grad_batch_size", type=int, default=4, 
                        help="Number of training examples to use for each gradient computation for swaps (default=4).")
    return parser.parse_args()

def load_dataset(dataset_path):
    with jsonlines.open(dataset_path, "r") as reader:
        dataset = [item for item in reader]
    return dataset

def make_list_of_tensor_dataset(str_dataset, tokenizer): 
    """ Given a dataset (type list[dict] each with keys "question", "answer", 
    "context", each a string) return a new list[dict] where each question_ids,
    answer_ids, and context_ids are [1, num_tokens] tensors tokenized using the
    model's tokenizer.  
    """
    tensor_dataset = []
    
    for item in tqdm(str_dataset):
        question = item["question"]
        answer = item["answer"]
        if "context" in item.keys(): 
            context = item["context"]
        else: 
            context = ""

        # if context is not null, prepend to question 
        if context != "" and context is not None:
            question = context + " Question: " + question + " Answer: "
        
        question_ids = tokenizer.encode(question, add_special_tokens=False, return_tensors="pt")
        answer_ids = tokenizer.encode(answer, add_special_tokens=False, return_tensors="pt")
        context_ids = tokenizer.encode(context, add_special_tokens=False, return_tensors="pt")
        
        tensor_item = {
            "question_ids": question_ids,
            "answer_ids": answer_ids,
            "context_ids": context_ids
        }
        
        tensor_dataset.append(tensor_item)
    
    return tensor_dataset

def optimize_prompt(dataset, out_dir, model, tokenizer, 
                    k, 
                    top_k, 
                    batch_size, 
                    num_iters, 
                    grad_batch_size, 
                    max_parallel=100):
    """
    Optimizes a prompt using the stochastic_easy_gcg_qa_ids function.

    Args:
        dataset (list[dict]): A list of dictionaries with keys "question_ids", "answer_ids", "context_ids".
            Each should be a list of tensors each of shape [1, num_tokens].
        out_dir (str): Path to the directory to save the optimized prompt and logs.
        model: Huggingface causal LLM.
        k (int): Length of the prompt to optimize.
        top_k (int): Number of top-k swaps to explore for each token in the prompt.
        batch_size (int): Batch size for exploring promising token swaps.
        num_iters (int): Number of iterations to run GCG for.
        grad_batch_size (int): Number of question-answer examples to aggregate gradients over before making swaps.

    Returns:
        prompt_ids (torch.Tensor): The optimized prompt tensor of shape [1, k].
    """
    question_ids = [item["question_ids"] for item in dataset]
    answer_ids = [item["answer_ids"] for item in dataset]

    # Call the stochastic_easy_gcg_qa_ids function to optimize the prompt
    prompt_ids, optim_hist = magic_words.stochastic_easy_gcg_qa_ids(
        question_ids=question_ids,
        answer_ids=answer_ids,
        num_tokens=k,
        model=model,
        tokenizer=tokenizer,
        top_k=top_k,
        batch_size=batch_size,
        num_iters=num_iters,
        grad_batch_size=grad_batch_size, 
        max_parallel=max_parallel
    )

    # Save the optimized prompt to a file
    prompt_file = os.path.join(out_dir, "optimized_prompt_text.txt")
    with open(prompt_file, "w") as f:
        prompt_text = tokenizer.decode(prompt_ids[0].tolist())
        f.write(prompt_text)

    # save the list of prompt_ids 
    prompt_ids_file = os.path.join(out_dir, "optimized_prompt_ids.json")
    with open(prompt_ids_file, "w") as f:
        json.dump(prompt_ids.tolist(), f)
    
    # save the optimization history 
    optim_hist_file = os.path.join(out_dir, "optimization_history.json")
    with open(optim_hist_file, "w") as f:
        json.dump(optim_hist, f)


    return prompt_ids


def main():
    args = parse_args()
    
    # Create the output directory if it doesn't exist
    # if out_dir is none, then just use today's date and time
    # from the datetime module
    if args.out_dir == "None": 
        args.out_dir = "results/"+datetime.datetime.now().strftime("%Y-%m-%d_at_%H-%M-%S")
    os.makedirs(args.out_dir, exist_ok=True)

    # dumping args dict to json file in out_dir
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # load model and tokenizer 
    print(f"Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # 16 bits 
    model.half()
    
    # Load the dataset
    print(f"Loading string dataset from {args.dataset}...")
    str_dataset = load_dataset(args.dataset)
    print("Done!")
    print(f"\nConverting to list of tensor dataset...")
    dataset = make_list_of_tensor_dataset(str_dataset, tokenizer)
    print("Done!")
    
    # Optimize the prompt using SGCG algorithm
    optimize_prompt(
        dataset=dataset,
        out_dir=args.out_dir,
        model=model,
        tokenizer=tokenizer,
        k=args.k,
        top_k=args.top_k,
        batch_size=args.batch_size,
        num_iters=args.num_iters,
        grad_batch_size=args.grad_batch_size, 
        max_parallel=args.max_parallel
    )

if __name__ == "__main__":
    main()