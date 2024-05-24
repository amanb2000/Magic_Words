import unittest
import sys 
import pdb 

import torch
import numpy as np
from tqdm import tqdm 

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from magic_words.forward_reach import _get_prompt_ids_brute_force
from magic_words.forward_reach import _get_answer_ids
from magic_words.forward_reach import _batch_get_answer_ids
from magic_words.forward_reach import get_reachable_gcg_set

from magic_words.greedy_forward import greedy_forward_reachability


class TestGreedyForward(unittest.TestCase): 
    @classmethod 
    def setUpClass(cls):
        """This is run once before all tests in this class. 
        """
        print("\n==========================================")
        print("=== Setting up TestGreedyForward class ===")
        print("==========================================")

        # Initialize a tokenizer and model
        cls.model_name = "tiiuae/falcon-7b"

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, padding_side="left")
        cls.tokenizer.pad_token = cls.tokenizer.eos_token

        cls.pipeline = transformers.pipeline(
            "text-generation",
            model=cls.model_name,
            tokenizer=cls.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        cls.model = cls.pipeline.model
        cls.model.eval()

        cls.device = cls.model.device
        print("Model device: ", cls.device)

    def setUp(self): 
        """This is run before every single individual test method. 
        """
        ... 
    def tearDown(self): 
        """This is run after every single individual test method. 
        """
        ...


    #####################
    ### MESSAGE TESTS ###
    #####################

    def test_brute_force(self): 
        prompt_ids = _get_prompt_ids_brute_force(self.tokenizer)
        print("Prompt_ids shape: ", prompt_ids.shape)

    def test_greedy_forward(self): 
        x_0 = "Hello, world!"
        greedy_forward_reachability(self.model, 
                                    self.tokenizer, 
                                    x_0=x_0, 
                                    max_prompt_length=5, 
                                    max_parallel=200,
                                    max_iters=100, 
                                    )
        