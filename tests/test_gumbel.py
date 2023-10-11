import unittest
import sys 
import pdb 

import torch
import numpy as np
from tqdm import tqdm 

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import magic_words
from magic_words import gumbel_prompt_hack, TauScheduler



class TestGumbelHack(unittest.TestCase): 
    @classmethod 
    def setUpClass(cls):
        """This is run once before all tests in this class. 
        """
        print("\n=======================================")
        print("=== Setting up TestGumbelHack class ===")
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


        cls.pipeline2 = transformers.pipeline(
            "text-generation",
            model=cls.model_name,
            tokenizer=cls.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        cls.model2 = cls.pipeline2.model
        cls.model2.eval()
        cls.model2.to(cls.device)

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
        print("Testing `gumbel_prompt_hack()` on the following string: ")
        future_str = "The squirrel and the fox had violent tendencies."
        print(f"\t{future_str}")

        scheduler = TauScheduler(init_tau=1.0, tau_dr=1-1e-3, T=500, min_tau=0.25, max_tau=2.0)
        m = 10
        max_iters = 10 # usually 10_000
        lr = 0.03
        batch_size = 256


        initial_loss, total_loss, message_fut_ids = gumbel_prompt_hack(self.model, 
                                    self.model2,
                                    self.tokenizer,
                                    future_str=future_str,
                                    m=m,
                                    batch_size=batch_size,
                                    scheduler=scheduler.exponentialDecay,
                                    max_iters=max_iters,
                                    lr=lr)


