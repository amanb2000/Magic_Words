import unittest
import sys 
import pdb 

import torch
import numpy as np
from tqdm import tqdm 

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import magic_words
from magic_words import greedy_prompt_hack_qa


class TestPromptHackQA(unittest.TestCase): 
    @classmethod 
    def setUpClass(cls):
        """This is run once before all tests in this class. 
        """
        print("\n=======================================")
        print("=== Setting up TestPromptHackQA class ===")
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
    def test_qa(self): 
        print("\nTesting the filler string functionality...")
        question_str = "Question: What does an ornithologist study? Answer: "
        answer_str = "Birds"
        num_tokens = 5

        search_limiter = magic_words.BruteForce(self.tokenizer.vocab_size)

        optimal_prompt, anti_logits = greedy_prompt_hack_qa(question_str, 
                                                        answer_str, 
                                                        num_tokens, 
                                                        self.model, 
                                                        self.tokenizer, 
                                                        search_limiter)

        # Todo: add assertion statement here
        print("Question: \t\t", question_str) 
        print("Desired answer: \t", answer_str)

        print("\nOptimal prompt: ", self.tokenizer.batch_decode(optimal_prompt))
        print("Optimal prompt ids: ", optimal_prompt)



        # Let's compare with ground truth!
        gnd_truth = torch.tensor([[12346, 29892, 22099, 29786, 34933]]).cpu()
        # corresponds to "Lincoln Importance President Title Facts"
        assert torch.allclose(optimal_prompt.cpu(), gnd_truth)
        print("Greedy prompt hack passed ground truth test!") 





