from magic_words.compute_score import compute_score
from magic_words.batch_compute_score import batch_compute_score
from magic_words.prompt_hack import greedy_prompt_hack
from magic_words.search_limiters import SearchLimiter, BruteForce
from magic_words.prompt_hack_qa import greedy_prompt_hack_qa, greedy_prompt_hack_qa_ids
from magic_words.backoff_hack_qa import backoff_hack_qa_ids

# Gumbel softmax
# from magic_words.gumbel_softmax_main import gumbel_prompt_hack, TauScheduler

# Greedy Coordinate Gradient
# from magic_words.coordinate_descent import gcg_prompt_hack
# from magic_words.coordinate_descent_qa import gcg_prompt_hack_qa
from magic_words.easy_gcg import easy_gcg_qa, easy_gcg_qa_ids

# Forward Generation 
from magic_words.forward_reach import get_reachable_set, forward_generate