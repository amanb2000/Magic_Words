import unittest
import sys 
import pdb 

import torch
import numpy as np
from tqdm import tqdm 

from magic_words import BruteForce


class TestSearchLimiter(unittest.TestCase): 
    @classmethod 
    def setUpClass(cls):
        """This is run once before all tests in this class. 
        """
        print("\n==========================================")
        print("=== Setting up TestSearchLimiter class ===")
        print("==========================================")

    
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

    def test_brute_force(self): 
        print("\nTesting BruteForce limiter...")
        vocab_size = 543
        limiter = BruteForce(vocab_size)
        candidates = limiter.get_candidates(3, 3)

        assert candidates.shape[0] == vocab_size
        assert candidates.shape[1] == 1
        print("Done.\n")

    def test_blacklist(self): 
        print("\nTesting BruteForce limiter with blacklist...")
        vocab_size = 543
        blacklist = [1, 2, 3, 4, 5]
        limiter = BruteForce(vocab_size, blacklist=blacklist)

        candidates = limiter.get_candidates(3, 3)
        assert candidates.shape[0] == vocab_size - len(blacklist)
        assert candidates.shape[1] == 1