""" Code for SearchLimiter abstract class and descendants. 

These objects are used to generate the candidate message_ids as we explore the 
prompt landscape. They narrow down (i.e., limit) the search to make it more 
efficient. 
"""

import torch 
import numpy as np
import pdb

from abc import ABC, abstractmethod


class SearchLimiter(ABC):
    """
    Abstract base class for limiting the search space in some way.
    Descendant classes must implement the get_candidates method.
    """
    
    def __init__(self, vocab_size):
        """
        Initialize SearchLimiter with vocab_size.

        Args:
            vocab_size (int): The size of the vocabulary.
        """
        self.vocab_size = vocab_size

    @abstractmethod
    def get_candidates(self, future_ids, future_mask, **kwargs):
        """
        Abstract method that needs to be implemented by descendant classes.
        
        Args:
            future_ids (Torch tensor): Tensor of future IDs.
            future_mask (Torch tensor): Tensor of the same shape as 
                `future_ids` with 1 for `future` and 0 for `prompt`. 
        """
        pass



class BruteForce(SearchLimiter):
    """
    A SearchLimiter descendant class that selects all vocabulary words 
    as candidates.
    """
    def __init__(self, vocab_size, blacklist=[]): 
        """
        vocab_size: int
            The size of the vocabulary.
        blacklist: list[int]
            A list of tokens to exclude from the search.
        """
        super().__init__(vocab_size)
        self.ids = torch.arange(vocab_size, dtype=torch.int64).unsqueeze(1)
        self.blacklist = blacklist

        if len(self.blacklist) > 0: 
            self.mask = torch.ones(self.vocab_size, dtype=torch.bool)
            self.mask[self.blacklist] = False
            self.ids = self.ids[self.mask]


    
    def get_candidates(self, future_ids, future_mask, **kwargs):
        """
        Returns a tensor containing all the vocabulary words as candidates.
        
        Args:
            future_ids (Torch tensor): Tensor of future IDs.
            future_mask (Torch tensor): Tensor of the same shape as 
                `future_ids` with 1 for `future` and 0 for `prompt`. 
        
        Returns:
            Torch tensor: A tensor with shape [vocab_size, 1].
        """
        return self.ids