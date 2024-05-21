# %%
"""
Load json from `train-v2.0.json` from https://rajpurkar.github.io/SQuAD-explorer/

We will only use questions where there is an actual answer. 
"""

# %%
import json 
import jsonlines as jsonl

# %%
dataset_path = 'train-v2.0.json'
# load with json
with open(dataset_path, 'r') as f:
    dataset = json.load(f)


# %%
print("Keys of raw dataset: \t", dataset.keys())
print("dataset['version']: \t", dataset['version'])
print("Data type of dataset['data']: \t", type(dataset['data']))
print("Length of dataset['data']: \t", len(dataset['data']))
print("Data type of dataset['data'][0]: \t", type(dataset['data'][0]))
print("Keys of dataset['data'][0]: \t", dataset['data'][0].keys())
print("Data type of dataset['data'][0]['title']: \t", type(dataset['data'][0]['title']))
print("dataset['data'][5]['title']: \t", dataset['data'][5]['title'])
print("Data type of data[5]['paragraphs']: \t", type(dataset['data'][0]['paragraphs']))
print("type of data[0]['paragraphs']: \t", type(dataset['data'][0]['paragraphs']))
print("Length of data[0]['paragraphs']: \t", len(dataset['data'][0]['paragraphs']))
print("Data type of data[0]['paragraphs'][0]: \t", type(dataset['data'][0]['paragraphs'][0]))
print("Keys of data[0]['paragraphs'][0]: \t", dataset['data'][0]['paragraphs'][0].keys())
print("dataset['data'][5]['paragraphs'][0]['qas'][0]", dataset['data'][5]['paragraphs'][0]['qas'][0])


# %% "question", "answer", "context", "title", and "id" 
# for all questions where is_impossible is False. 

import pandas as pd

# Initialize an empty list to store the data
data = []

# Iterate over the dataset
for item in dataset['data']:
    title = item['title']
    for paragraph in item['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            if not qa['is_impossible']:
                question = qa['question']
                answer = qa['answers'][0]['text']
                answer_start = qa['answers'][0]['answer_start']
                qa_id = qa['id']
                
                # Append the data to the list
                data.append({
                    'question': question,
                    'answer': answer,
                    'context': context,
                    'title': title,
                    'id': qa_id,
                    'answer_start': answer_start
                })

# %% Create a DataFrame from the data list
df = pd.DataFrame(data)
print("Done making dataframe. ")
print("Shape of the DataFrame: ", df.shape)

# Print the first few rows of the DataFrame
print(df.head())

# %% 
# save to jsonl 
print("Saving to jsonl at 'squad_train_v2.0.jsonl'")
df.to_json('squad_train_v2.0.jsonl', orient='records', lines=True)
print("Done!")

# %%
