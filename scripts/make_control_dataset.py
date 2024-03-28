import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import random
import pdb

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a control dataset of Q&A pairs.")
    parser.add_argument("--out_file", type=str, default="control_dataset.csv", help="Output file name")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name for tokenizer")
    parser.add_argument("--question_lens", type=int, nargs='+', required=True, help="List of question lengths in tokens")
    parser.add_argument("--answer_lens", type=int, nargs='+', required=True, help="List of answer lengths in tokens")
    parser.add_argument("--num_examples_per_qa_size", type=int, required=True, help="Number of examples per Q&A size")

    # print args prettily 
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print()

    return args

def generate_qa_pairs(tokenizer, dataset, question_lens, answer_lens, num_examples):
    qa_pairs = []
    
    for q_len in question_lens:
        for a_len in answer_lens:
            total_len = q_len + a_len
            num_generated = 0
            while num_generated < num_examples:
                # Randomly select a sample from the dataset
                sample = random.choice(dataset['train'])
                text = sample['text']
                # Tokenize the text
                tokens = tokenizer(text)['input_ids']

                # If the text is shorter than the required total length, skip
                if len(tokens) < total_len*4:
                    continue
                else: 
                    num_generated+=1
                
                # Randomly select a start index
                start_index = random.randint(0, len(tokens) - total_len)
                end_index = start_index + total_len
                
                # Split tokens into question and answer
                question_ids = tokens[start_index:start_index + q_len]
                answer_ids = tokens[start_index + q_len:end_index]
                
                # Convert tokens to strings
                question = tokenizer.decode(question_ids)
                answer = tokenizer.decode(answer_ids)
                
                qa_pairs.append([question, answer, question_ids, answer_ids])
                
    return qa_pairs

def main():
    args = parse_arguments()

    # Load the dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # ensure no extra tokens are added


    # Generate Q&A pairs
    qa_pairs = generate_qa_pairs(tokenizer, dataset, args.question_lens, args.answer_lens, args.num_examples_per_qa_size)

    # Save to CSV
    df = pd.DataFrame(qa_pairs, columns=["question", "answer", "question_ids", "answer_ids"])
    df.to_csv(args.out_file, index=False)
    
    print("Control dataset generated successfully.")

if __name__ == "__main__":
    main()
