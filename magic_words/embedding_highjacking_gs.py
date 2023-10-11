# In this file, we will create a class where one can instantiate an LLM, with embeddings as trainable parameters.


# This was the most annoying this to create...

import torch
import torch.nn as nn
import pdb

class WordEmbeddingOverride(nn.Module):
    def __init__(self, model, m, device, future_embedding_vectors, embedding_dict, init_noise=0.1, init_tau=2.0):
        super(WordEmbeddingOverride, self).__init__()
        self.model = model
        self.m = m
        self.device = device

        d_model = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        self.batch_size = future_embedding_vectors.shape[0]

        self.embedding_dict = embedding_dict

        # Select batch_size x m random vectors from embedding_dict
        # First, select batch_size x m random integers from 0 to vocab_size
        # Then, select the corresponding vectors from embedding_dict
        # Finally, reshape to batch_size x m x d_model
        gumbel_logits = torch.zeros((1, m, self.vocab_size), device=self.device) + init_noise*torch.randn((1, m, self.vocab_size), device=self.device)
        #embeddings_init = embedding_dict[torch.randint(0, vocab_size, (batch_size, m))].reshape(batch_size, m, d_model)

        #self.embeddings_trainable = nn.Parameter(embeddings_init)
        self.logits_trainable = nn.Parameter(gumbel_logits)
        self.future_embedding_vectors = future_embedding_vectors
        self.tau = init_tau

    def forward(self, input_ids, **kwargs):

        # Input_ids doesn't do anything

        # Concatenate the embedding message with the future embeddings

        
        uniform_noise = torch.rand((self.batch_size, self.m, self.vocab_size), device=self.device)
        g_i = -torch.log(-torch.log((1 - 1e-9)*uniform_noise) + 1e-9)

        gumbel_softmax = nn.functional.softmax((self.logits_trainable + g_i)/self.tau, dim=-1) # [batch_size, m, vocab_size]

        superposition_embeddings = torch.einsum("bmv,vd->bmd", gumbel_softmax, self.embedding_dict.float())

        y = torch.cat((superposition_embeddings.bfloat16(), self.future_embedding_vectors.detach()), dim=1)

        return y
    
    def update_tau(self, tau):
        self.tau = tau

class LLMEmbeddingHighjacking(nn.Module):
    def __init__(self, model, m, device, future_token_ids):
        super(LLMEmbeddingHighjacking, self).__init__()
        self.model = model
        self.device = device
        self.m = m
        self.future_length = future_token_ids.shape[1]
        self.batch_size = future_token_ids.shape[0]
        d_model = model.config.hidden_size
        vocab_size = model.config.vocab_size

        # Extract the word embedding vectors from the model [vocab_size, d_model]
        self.embedding_word = model.transformer.word_embeddings.weight.detach().clone().to(self.device)

        # Get the future embedding vectors
        future_embedding_vectors = self.embedding_word[future_token_ids]

        self.weo = WordEmbeddingOverride(model, m, device, future_embedding_vectors, self.embedding_word)

        # Set the model's embedding layer to just pass the vectors through
        self.model.transformer.word_embeddings = self.weo

    def forward(self, **kwargs):

        input_ids = torch.zeros(self.batch_size, self.m + self.future_length, dtype=torch.long, device=self.device)

        y = self.model(input_ids, **kwargs)

        return y
    
    def update_tau(self, tau):
        self.weo.update_tau(tau)

    def get_best_message_ids(self):
        # Get the best message ids
        # Get the logits
        logits = self.weo.logits_trainable
        # Get the best message ids
        best_message_ids = torch.argmax(logits, dim=-1)
        return best_message_ids
    
    def get_sample_message_ids(self):
        # Get the logits
        logits = self.weo.logits_trainable
        # Now sample from the categorical distribution implied
        sample_message_ids = torch.distributions.Categorical(logits=logits).sample()
        return sample_message_ids