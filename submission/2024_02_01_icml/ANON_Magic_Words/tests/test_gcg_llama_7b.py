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

class TestGCGLlama7b(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """This is run once before all tests in this class.
        """
        print("\n=======================================")
        print("==== Setting up TestGCGLlama7b class =====")
        print("=======================================")

        # Initialize a tokenizer and model
        QUANTIZED = False
        if QUANTIZED:
            cls.model_name = "TheBloke/Llama-2-7B-GPTQ"
            cls.model_basename = "model"
            cls.use_triton = False

            cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, use_fast=True)

            cls.model = AutoGPTQForCausalLM.from_quantized(cls.model_name,
                model_basename=cls.model_basename,
                inject_fused_attention=False, # Required for Llama 2 70B model at this time.
                use_safetensors=True,
                trust_remote_code=False,
                device="cuda:0",
                use_triton=cls.use_triton,
                quantize_config=None)
            cls.model = exllama_set_max_input_length(cls.model, 4096)
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
        future_str = "The squirrel and the fox had violent tendencies."
        print(f"\t{future_str}")

        m = 10
        max_iters = 1000 # usually 10_000
        k = 32
        keep_best=False
        batch_size = 10
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


