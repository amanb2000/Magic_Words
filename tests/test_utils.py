import unittest
import sys 
import pdb 

import torch
import numpy as np
from tqdm import tqdm 

from magic_words import utils


class TestUtils(unittest.TestCase): 
    @classmethod 
    def setUpClass(cls):
        """This is run once before all tests in this class. 
        """
        print("\n==================================")
        print("=== Setting up TestUtils class ===")
        print("==================================")


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


    def test_concatenation(self): 
        print("Testing `cat_msg_future()` function...")

        message_ids = torch.Tensor([[2, 3, 4], [4, 3, 2]]) #[batch=2, msg_toks=3]
        future_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        insertion_point=1

        msg_fut = utils.cat_msg_future(message_ids, 
                                    future_ids,
                                    insertion_point)

        gnd_truth = torch.Tensor([[1, 2, 3, 4, 2, 3, 4, 5, 6], 
                                  [1, 4, 3, 2, 2, 3, 4, 5, 6]])

        assert torch.allclose(msg_fut, gnd_truth)

    def test_zero_concatenation(self): 
        print("Testing `cat_msg_future()` with prepending (insertion_point=0)")
        
        message_ids = torch.Tensor([[2, 3, 4], [4, 3, 2]]) #[batch=2, msg_toks=3]
        future_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        insertion_point=0

        msg_fut = utils.cat_msg_future(message_ids, 
                                    future_ids,
                                    insertion_point)

        gnd_truth = torch.Tensor([[2, 3, 4, 1, 2, 3, 4, 5, 6], 
                                  [4, 3, 2, 1, 2, 3, 4, 5, 6]])

        # pdb.set_trace()

        assert torch.allclose(msg_fut, gnd_truth)
