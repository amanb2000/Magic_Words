import unittest
import sys 
import pdb 

import torch
import numpy as np
from tqdm import tqdm 

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import magic_words
from magic_words import gcg_prompt_hack

class TestGCGHack(unittest.TestCase): 
    @classmethod 
    def setUpClass(cls):
        """This is run once before all tests in this class. 
        """
        print("\n=======================================")
        print("==== Setting up TestGCGHack class =====")
        print("=======================================")

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
        print("Testing `gcg_prompt_hack()` on the following string: ")
        future_str = "The squirrel and the fox had violent tendencies."
        print(f"\t{future_str}")

        m = 10
        max_iters = 1000 # usually 10_000
        k = 32
        keep_best=False
        batch_size = 256
        Temp = 1.0

        initial_loss, final_loss, message_fut_ids = gcg_prompt_hack(self.model,
                    self.tokenizer,
                    future_str,
                    m=m,
                    batch_size=batch_size,
                    k=k,
                    keep_best=keep_best,
                    temp=Temp,
                    max_iters=max_iters)




