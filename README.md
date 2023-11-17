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
python3 scripts/backoff_hack_demo.py
```
See the comments in the script for further details. [This issue
thread](https://github.com/amanb2000/Magic_Words/blob/1986861b51433fb7ad55ef39cde98afd1d76535c/scripts/backoff_hack_demo.py#L113)
is also a good resource for getting up and running.


## Testing
```
# run all tests: 
coverage run -m unittest discover

# get coverage report:
coverage report --include=prompt_landscapes/*

# run a specific test:
coverage run -m unittest tests/test_compute_score.py
``````




