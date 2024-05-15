import unittest
import sys 
import pdb 

import torch
import numpy as np
from tqdm import tqdm 

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq import exllama_set_max_input_length

import magic_words
from magic_words import gcg_prompt_hack

class TestLlama7b(unittest.TestCase): 
    @classmethod 
    def setUpClass(cls):
        """This is run once before all tests in this class. 
        """
        print("\n=======================================")
        print("==== Setting up TestLlama7b class =====")
        print("=======================================")

        # Initialize a tokenizer and model
        # Load model directly
        QUANTIZED=True
        if QUANTIZED: 
            cls.model_name = "TheBloke/Llama-2-7B-GPTQ"
            # cls.model_name = "TheBloke/Llama-2-13B-GPTQ"
            cls.model_basename = "model"
            cls.use_triton = False

            cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, use_fast=True)

            cls.model = AutoGPTQForCausalLM.from_quantized(cls.model_name,
                model_basename=cls.model_basename,
                inject_fused_attention=False, # Required for Llama 2 70B model at this time.
                use_safetensors=True,
                trust_remote_code=False,
                device_map='auto',
                use_triton=cls.use_triton,
                quantize_config=None)
            # cls.model = exllama_set_max_input_length(cls.model, 16384)
            cls.model.eval()

            cls.device = cls.model.device
        else:
            cls.model_name = "huggyllama/llama-7b"
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
            cls.model = AutoModelForCausalLM.from_pretrained(cls.model_name, device_map="auto")

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

    def test_greedy(self): 
        print("MODEL NAME: ", self.model_name)
        print("Testing `gcg_prompt_hack()` on the following string: ")
        future_str = "The squirrel and the fox had violent tendencies."
        print(f"\t{future_str}")

        num_prompt_tokens = 2
        keep_best=False
        batch_size = 256
        Temp = 1.0

        search_limiter = magic_words.BruteForce(self.tokenizer.vocab_size)

        prompt, scores = magic_words.greedy_prompt_hack(future_str, 
                       num_prompt_tokens, 
                       self.model,
                       self.tokenizer,
                       search_limiter,
                       max_parallel=511)

        # pdb.set_trace()
        print("Prompt: ", self.tokenizer.batch_decode(prompt))



