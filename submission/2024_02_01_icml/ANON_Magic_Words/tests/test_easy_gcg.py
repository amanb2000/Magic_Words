import unittest
import sys 
import pdb 

import torch
import numpy as np
from tqdm import tqdm 

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import magic_words
from magic_words import easy_gcg_qa, easy_gcg_qa_ids

class TestEasyGCG(unittest.TestCase): 
    @classmethod 
    def setUpClass(cls):
        """This is run once before all tests in this class. 
        """
        print("\n=======================================")
        print("==== Setting up TestEasyGCG class =====")
        print("=======================================")

        # Initialize a tokenizer and model
        cls.model_name = "tiiuae/falcon-40b"

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
        """ This is run before every single individual test method.
        """
        ...

    def tearDown(self): 
        """This is run after every single individual test method. 
        """
        ...


    #####################
    ### MESSAGE TESTS ###
    #####################

    def test_run(self): 
        print("Testing `easy_gcg()` on a simple example.")
        question_str = "What is the meaning of life? "
        answer_str = "42"

        num_tokens = 10
        top_k = 128
        max_iters = 34 
        batch_size = 768
        max_parallel = 101

        prompt_ids = easy_gcg_qa(question_str,
                                answer_str,
                                num_tokens,
                                self.model,
                                self.tokenizer,
                                top_k,
                                batch_size=batch_size,
                                num_iters=max_iters,
                                max_parallel=max_parallel,
                                blacklist=[]) # just to test the blacklist

        print("Best prompt: ", prompt_ids)
        print("Decoded prompt: ", self.tokenizer.batch_decode(prompt_ids))
        







