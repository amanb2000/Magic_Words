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


class TestForwardReach(unittest.TestCase): 
    @classmethod 
    def setUpClass(cls):
        """This is run once before all tests in this class. 
        """
        print("\n=============================================")
        print("=== Setting up TestScoreComputation class ===")
        print("=============================================")

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
        print("Answer_ids: ", answer_ids)
