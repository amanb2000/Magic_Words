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

## Testing
```
# run all tests: 
coverage run -m unittest discover

# get coverage report:
coverage report --include=prompt_landscapes/*

# run a specific test:
coverage run -m unittest tests/test_compute_score.py
``````




