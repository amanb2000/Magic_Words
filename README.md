# Magic_Words

Code for the paper [What's the Magic Word? A Control Theory of LLM Prompting](https://arxiv.org/abs/2310.04444).

Implements greedy back generation and greedy coordinate gradient (GCG) to find 
optimal control prompts (_magic words_). 

## Setup

```
# create a virtual environment
python3 -m venv venv

# activate the virtual environment
source venv/bin/activate

# install the package and dependencies
pip install -e .
pip install -r requirements.txt
```

## Example Script

Run the script in `scripts/backoff_hack.py` for a demo of finding the **magic
words** (optimal control prompt) for a given question-answer pair using greedy
search and greedy coordinate gradient (GCG). It applies the same algorithms as
in the [LLM Control Theory](https://arxiv.org/abs/2310.04444) paper: 

```
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
```


## Testing
```
# run all tests: 
coverage run -m unittest discover

# get coverage report:
coverage report --include=prompt_landscapes/*

# run a specific test:
coverage run -m unittest tests/test_compute_score.py
``````




