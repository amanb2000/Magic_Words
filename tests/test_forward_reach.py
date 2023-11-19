import unittest
import sys 
import pdb 

import torch
import numpy as np
from tqdm import tqdm 

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from magic_words.forward_reach import _get_prompt_ids_brute_force


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

        # cls.pipeline = transformers.pipeline(
        #     "text-generation",
        #     model=cls.model_name,
        #     tokenizer=cls.tokenizer,
        #     torch_dtype=torch.bfloat16,
        #     trust_remote_code=True,
        #     device_map="auto",
        # )

        # cls.model = cls.pipeline.model
        # cls.model.eval()

        # cls.device = cls.model.device
        # print("Model device: ", cls.device)

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
        pdb.set_trace()
        print("Prompt_ids shape: ", prompt_ids.shape)
