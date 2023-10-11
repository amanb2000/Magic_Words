import unittest
import sys 
import pdb 

import torch
import numpy as np
from tqdm import tqdm 

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from magic_words import compute_score


class TestScoreComputation(unittest.TestCase): 
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

    def test_run(self): 
        print("\nTesting the compute_score function...")
        message_ids_np = np.zeros([self.tokenizer.vocab_size], dtype=np.int64)
        message_string_list = []

        # test every single possible 1-token message
        for i in range(self.tokenizer.vocab_size):
            msg = self.tokenizer.decode([i])
            message_ids_np[i] = i
            message_string_list.append(msg)
        
        max_parallel = 100

        message_string_list = message_string_list[:max_parallel]
        message_ids_np = message_ids_np[:max_parallel]

        future = 'ello world.'
        future_ids = self.tokenizer.encode(future, return_tensors='pt')
        # future_ids has shape [1, num_future_tokens]

        messages_ids = torch.Tensor(message_ids_np)
        messages_ids = messages_ids.int()
        messages_ids = messages_ids.unsqueeze(1)
        # messsages_ids has shape [num_messages, num_message_tokens]
            # in this case, num_message_tokens = 1

        losses, mean_total_loss = compute_score(messages_ids, future_ids, self.model, self.tokenizer)

        assert type(losses) == torch.Tensor

        for i in range(len(message_string_list)):
            print("Message: ", message_string_list[i], " Loss: ", losses[i].item())

        print("Compute_score() ran successfully")


    def test_common_sense(self): 
        print("\nTesting the compute_score function for common sense values...")
        # let see if message='h' for 'ello world' is the highest scoring message. 

        message_strings = ['h', 
                           'the next words are "hello world": h', 
                           'hello world, hello world, h']
        future = 'ello world.'

        message_ids = self.tokenizer.batch_encode_plus(message_strings, return_tensors='pt', padding='longest')['input_ids']

        future_ids = self.tokenizer.encode(future, return_tensors='pt')
        # future_ids has shape [1, num_future_tokens]

        losses, mean_total_loss = compute_score(message_ids, future_ids, self.model, self.tokenizer)

        assert type(losses) == torch.Tensor

        for i in range(len(message_strings)): 
            print("Message: ", message_strings[i], " Loss: ", losses[i].item())
