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


class TestForwardReach(unittest.TestCase): 
    @classmethod 
    def setUpClass(cls):
        """This is run once before all tests in this class. 
        """
        print("\n=========================================")
        print("=== Setting up TestForwardReach class ===")
        print("=========================================")

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

    def test_get_answer(self): 
        question = "What is the meaning of life?"
        question_ids = self.tokenizer(question, return_tensors="pt")["input_ids"]

        prompt_ids = _get_prompt_ids_brute_force(self.tokenizer)

        # select the first 300 
        prompt_ids = prompt_ids[:300, :]

        answer_ids = _get_answer_ids(prompt_ids, question_ids, self.model, self.tokenizer)

        print("Answer_ids shape: ", answer_ids.shape)
        # print("Answer_ids: ", answer_ids)

    def test_batch_get_answer(self):
        question = "What is the meaning of life?"
        question_ids = self.tokenizer(question, return_tensors="pt")["input_ids"]

        prompt_ids = _get_prompt_ids_brute_force(self.tokenizer)

        # select the first 300
        prompt_ids = prompt_ids[:300, :]
        max_parallel = 100

        answer_ids_batch = _batch_get_answer_ids(prompt_ids, question_ids, 
                                           self.model, 
                                           self.tokenizer, 
                                           max_parallel=max_parallel)
        
        # now make sure they're the same as before 
        answer_ids = _get_answer_ids(prompt_ids, question_ids, self.model, self.tokenizer)

        print("Answer_ids_batch shape: ", answer_ids_batch.shape)
        print("Answer_ids_batch: ", answer_ids_batch)

        # torch allclose 
        self.assertTrue(torch.allclose(answer_ids, answer_ids_batch, atol=1e-5))
        print("Passed allclose!")

    def test_get_reachable_gcg_set(self): 
        x_0 = "What is the meaning of life?"
        x_0_ids = self.tokenizer(x_0, return_tensors="pt")["input_ids"]

        top_k = 128
        num_prompt_tokens = 100
        batch_size = 768
        num_iters = 34
        max_parallel = 300
        num_init_prompts = 1
        num_to_mutate = 10
        eps_e = 0.0

        reachable_gcg_set = get_reachable_gcg_set(x_0_ids, self.model, self.tokenizer, 
                                                  top_k=top_k,
                                                  num_prompt_tokens=num_prompt_tokens,
                                                  batch_size=batch_size,
                                                  num_iters=num_iters,
                                                  max_parallel=max_parallel,
                                                  num_init_prompts=num_init_prompts,
                                                  num_to_mutate=num_to_mutate, 
                                                  eps_e=eps_e)
        pdb.set_trace()
        print("Reachable_gcg_set shape: ", reachable_gcg_set.shape)
        print("Reachable_gcg_set: ", reachable_gcg_set)
        pdb.set_trace()

