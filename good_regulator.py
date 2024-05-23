import argparse
import json
import os
import random
from tqdm import tqdm
from openai import OpenAI

import pdb

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize OpenAI API key

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def evaluate_prompt(prompt, dataset, model_name):
    correct = 0
    total = 0
    examples = []
    
    for entry in dataset:
        question = entry['question']
        answer = entry['answer']
        input_text = prompt.format(question)
        
        response = client.chat.completions.create(model=model_name,
        messages=[{"role": "user", "content": input_text}],
        max_tokens=1,
        n=1,
        stop=None)

        # response = client.chat.completions.create(model="gpt-4o", messages= [{"role": "user", "content": prompt }])

        prediction = response.choices[0].message.content.strip().upper()
        if prediction == answer:
            correct += 1
        examples.append({
            "question": question,
            "predicted": prediction,
            "actual": answer
        })
        total += 1

    accuracy = correct / total
    return accuracy, examples

def generate_new_prompts(metaprompt, examples, current_prompt, optimizer_model, N):
    input_text = metaprompt.format(current_prompt, json.dumps(examples))
    response = openai.ChatCompletion.create(
        model=optimizer_model,
        messages=[{"role": "user", "content": input_text}],
        max_tokens=300,
        n=N,
        stop=None
    )
    return [choice['message']['content'].strip() for choice in response['choices']]


def main(args):
    dataset = load_jsonl(args.dataset)
    with open(args.init_metaprompt, 'r') as f:
        metaprompt = f.read()
    with open(args.init_prompt, 'r') as f:
        current_prompt = f.read()

    results = []
    for iteration in tqdm(range(args.num_iters), desc="Iterations"):
        accuracy, examples = evaluate_prompt(current_prompt, dataset, args.model_name)
        results.append({
            "iteration": iteration,
            "prompt": current_prompt,
            "accuracy": accuracy,
            "examples": examples
        })
        
        if len(current_prompt) > args.k:
            current_prompt = current_prompt[:args.k]
        
        new_prompts = generate_new_prompts(metaprompt, examples, current_prompt, args.optimizer_model, args.N)
        new_prompt_accuracies = [(prompt, evaluate_prompt(prompt, dataset, args.model_name)[0]) for prompt in new_prompts]
        new_prompt_accuracies.sort(key=lambda x: x[1], reverse=True)
        
        current_prompt = new_prompt_accuracies[0][0]  # Select the best new prompt

    os.makedirs(args.out_dir, exist_ok=True)
    save_json(results, os.path.join(args.out_dir, "optimization_results.json"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-based prompt optimization for medical data annotation.")
    parser.add_argument("--dataset", type=str, default="datasets/akul_train_set_prompting.jsonl", help="Path to the dataset file.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to store results.")
    parser.add_argument("--model_name", type=str, default="garage-bAInd/Platypus2-70B-instruct", help="Model name for evaluation.")
    parser.add_argument("--optimizer_model", type=str, default="Llama-3-70b-Instruct", help="Model name for generating new prompts.")
    parser.add_argument("--init_metaprompt", type=str, default="prompts/init_akul_metaprompt.txt", help="Path to the initial metaprompt file.")
    parser.add_argument("--init_prompt", type=str, default="prompts/init_akul_template.txt", help="Path to the initial prompt file.")
    parser.add_argument("--k", type=int, default=300, help="Maximum number of characters in the prompt.")
    parser.add_argument("--N", type=int, default=30, help="Number of alternative prompts to generate each iteration.")
    parser.add_argument("--num_iters", type=int, default=100, help="Number of optimization iterations.")
    args = parser.parse_args()
    main(args)

