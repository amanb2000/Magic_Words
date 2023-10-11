# In this file, we will create a class where one can instantiate an LLM, with embeddings as trainable parameters.


# This was the most annoying this to create...

import torch
import torch.nn as nn
import pdb

import auto_gptq
import transformers

class WordEmbeddingOverride(nn.Module):
    def __init__(self, model, m, device, future_embedding_vectors, embedding_dict):
        super(WordEmbeddingOverride, self).__init__()
        self.model = model
        self.m = m
        self.device = device

        d_model = model.config.hidden_size
        vocab_size = model.config.vocab_size
        batch_size = future_embedding_vectors.shape[0]

        # Select batch_size x m random vectors from embedding_dict
        # First, select batch_size x m random integers from 0 to vocab_size
        # Then, select the corresponding vectors from embedding_dict
        # Finally, reshape to batch_size x m x d_model
        embeddings_init = embedding_dict[torch.randint(0, vocab_size, (batch_size, m))].reshape(batch_size, m, d_model)


        self.embeddings_trainable = nn.Parameter(embeddings_init)
        self.future_embedding_vectors = future_embedding_vectors


    def forward(self, input_ids, **kwargs):

        # Input_ids doesn't do anything

        # Concatenate the embedding message with the future embeddings
        
        # TODO: Add control flow for casting self.embeddings_trainable 
        # if we are using a falcon model!
        # y = torch.cat((self.embeddings_trainable.bfloat16(), self.future_embedding_vectors.detach()), dim=1)
        y = torch.cat((self.embeddings_trainable, self.future_embedding_vectors.detach()), dim=1)

        return y

    def update_embeddins_trainable(self, embeddings_trainable):
        self.embeddings_trainable = nn.Parameter(embeddings_trainable)

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

        # Get the future embedding vectors
        if type(model) == auto_gptq.modeling.llama.LlamaGPTQForCausalLM:
            # Here we are dealing with a quantized llama model. 
            future_embedding_vectors = model.model.model.embed_tokens(future_token_ids)
            # Extract the word embedding vectors from the model [vocab_size, d_model]
            self.embedding_word = model.model.model.embed_tokens.weight.detach().clone().to(self.device)
        elif type(model) == transformers.models.llama.modeling_llama.LlamaForCausalLM: 
            # Here we are dealing with a non-quantized Llama model.
            future_embedding_vectors = model.model.embed_tokens(future_token_ids)
            # Extract the word embedding vectors from the model [vocab_size, d_model]
            self.embedding_word = model.model.embed_tokens.weight.detach().clone().to(self.device)
        else:
            # here we are dealing with a Falcon model :) 
            future_embedding_vectors = model.transformer.word_embeddings(future_token_ids)
            # Extract the word embedding vectors from the model [vocab_size, d_model]
            self.embedding_word = model.transformer.word_embeddings.weight.detach().clone().to(self.device)


        self.weo = WordEmbeddingOverride(model, m, device, future_embedding_vectors, self.embedding_word)

        # Set the model's embedding layer to just pass the vectors through
        if type(model) == auto_gptq.modeling.llama.LlamaGPTQForCausalLM:
            self.model.model.model.embed_tokens = self.weo
        elif type(model) == transformers.models.llama.modeling_llama.LlamaForCausalLM: 
            self.model.model.embed_tokens = self.weo
        else:
            self.model.transformer.word_embeddings = self.weo

    def forward(self, **kwargs):

        input_ids = torch.zeros(self.batch_size, self.m + self.future_length, dtype=torch.long, device=self.device)


        y = self.model(input_ids, **kwargs)

        return y
    
    def update_embedding_message(self, embedding_message):
        self.weo.update_embeddins_trainable(embedding_message)