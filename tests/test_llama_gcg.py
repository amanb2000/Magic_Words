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
from magic_words import gcg_prompt_hack, gcg_prompt_hack_qa

class TestLlamaGCGHack(unittest.TestCase): 
    @classmethod 
    def setUpClass(cls):
        """This is run once before all tests in this class. 
        """
        print("\n============================================")
        print("==== Setting up TestLlamaGCGHack class =====")
        print("============================================")
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

    def test_run(self): 
        print("Testing `gcg_prompt_hack()` on the following string: ")
        question_str = "The squirrel and the fox had violent tendencies."
        answer_str = "Oh"
        print(f"\t{question_str}")

        m = 10
        max_iters = 1000 # usually 10_000
        k = 64
        keep_best=False
        batch_size = 16
        Temp = 1.0

        initial_loss, final_loss, best_message = gcg_prompt_hack_qa(self.model,
                self.tokenizer,
                question_str,
                answer_str,
                m=m,                    # Number of tokens in message
                batch_size=batch_size,         # Batch size for gradient computation
                batches_before_swap=8,  # Number of batches before swapping in new message
                k=k,                   # Size of candidate replacement set for each token in message
                keep_best=keep_best,        # Keeps best message each iteration (often leads to premature convergence)
                temp=Temp,               # Temperature for ce loss when deciding which token to replace
                max_iters=max_iters,
                blacklist=[0, 1, 2, 3])         # Number of iterations to run for
