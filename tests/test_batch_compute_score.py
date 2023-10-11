import unittest
import sys 
import pdb 

import torch
import numpy as np
from tqdm import tqdm 

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from magic_words import compute_score, batch_compute_score


class TestBatchComputeScore(unittest.TestCase): 
    @classmethod 
    def setUpClass(cls):
        """This is run once before all tests in this class. 
        """
        print("\n==============================================")
        print("=== Setting up TestBatchComputeScore class ===")
        print("==============================================")

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

    def test_all_messages(self): 
        print("\nTesting the batch_compute_score() function on all possible messages...")

        future_str = ": The greatest to ever do it."
        future_ids = self.tokenizer.encode(future_str, return_tensors="pt").to(self.device)

        # Now let's synthesize the `message_ids` -- it should be a torch 
        # tensor of shape [num_messages, 1] == [num_messages, len_message]. 
        vocab_size = self.tokenizer.vocab_size
        message_ids = torch.arange(vocab_size, 
                                   dtype=torch.int64,
                                   device=self.model.device).unsqueeze(1)

        message_scores = batch_compute_score(message_ids, 
                                             future_ids, 
                                             self.model, 
                                             self.tokenizer, 
                                             max_parallel=1000)

        print(f"Argmin for test string `{future_str}`: {message_scores.argmin()}, corresponds to token: `{self.tokenizer.decode(message_scores.argmin())}`")
        print(f"\tThe CE loss associated with this message was {message_scores.min()}")



    def test_subset(self): 
        print("\nTesting batch_compute_score() on a smaller subset of messages...")
        
        future_str = ": The greatest to ever do it."
        future_ids = self.tokenizer.encode(future_str, return_tensors="pt").to(self.device)

        # Now let's synthesize the `message_ids` -- it should be a torch 
        # tensor of shape [num_messages, 1] == [num_messages, len_message]. 
        vocab_size = self.tokenizer.vocab_size // 8
        print(f"Surveying first {vocab_size} of {self.tokenizer.vocab_size} tokens only")
        message_ids = torch.arange(vocab_size, 
                                   dtype=torch.int64,
                                   device=self.model.device).unsqueeze(1)

        message_scores = batch_compute_score(message_ids, 
                                             future_ids, 
                                             self.model, 
                                             self.tokenizer, 
                                             max_parallel=1000)

        print(f"Argmin for test string `{future_str}`: {message_scores.argmin()}, corresponds to token: `{self.tokenizer.decode(message_scores.argmin())}`")
        print(f"\tThe CE loss associated with this message was {message_scores.min()}")


    def test_monoset(self): 
        print("\nLet's make sure the system still works when there is less than 1 batch worth of candidates")

        future_str = ": The greatest to ever do it."
        future_ids = self.tokenizer.encode(future_str, return_tensors="pt").to(self.device)

        # Now let's synthesize the `message_ids` -- it should be a torch 
        # tensor of shape [num_messages, 1] == [num_messages, len_message]. 
        vocab_size = self.tokenizer.vocab_size // 1000
        print(f"Surveying first {vocab_size} of {self.tokenizer.vocab_size} tokens only")
        message_ids = torch.arange(vocab_size, 
                                   dtype=torch.int64,
                                   device=self.model.device).unsqueeze(1)

        message_scores = batch_compute_score(message_ids, 
                                             future_ids, 
                                             self.model, 
                                             self.tokenizer, 
                                             max_parallel=1000)

        print(f"Argmin for test string `{future_str}`: {message_scores.argmin()}, corresponds to token: `{self.tokenizer.decode(message_scores.argmin())}`")
        print(f"\tThe CE loss associated with this message was {message_scores.min()}")


