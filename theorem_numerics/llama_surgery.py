""" Script version of the theorem_validation.ipynb notebook
"""
# Import box 
import torch 
import numpy 
import transformers 
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
import pdb

model_name = "meta-llama/Meta-Llama-3-8B"
skip_norm = False
layer_num = 3

print(f"\nLoading model {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# if on CUDA 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("Done!\n")


print("\nDefining input x_0, running thru model without grad...")
x_0 = "I am become death, destroyer of worlds."
x_0_ids = tokenizer.encode(x_0, return_tensors="pt").to(device)
# get the activations 
with torch.no_grad():
    # skip_norm=True triggers custom debug code to prevent the final normalization 
    # layer that norms over all the hidden states. This will let us compare the 
    # output of a single layer with the activations in the model. 
    outputs = model(x_0_ids, output_hidden_states=True, skip_norm=skip_norm)
print("Done! Outputs keys: ", outputs.keys())



print("Grabbing a single attention head, running some values thru it...")
def get_single_attn_head(model, layer_num, return_layer=False): 
    """ Given a Llama-3 type model, get the layer_num attention head. 

    Decoder layers are called `class LlamaDecoderLayer`, attention heads 
    are `class LlamaAttention`

    """
    assert type(model) == transformers.models.llama.modeling_llama.LlamaForCausalLM, "Model must be a Llama model."
    assert layer_num < len(model.model.layers), "Layer number out of bounds"
    if not return_layer: 
        return model.model.layers[layer_num].self_attn
    else: 
        return model.model.layers[layer_num].self_attn, model.model.layers[layer_num]

print(f"Grabbing a single attention head, layer from idx = {layer_num}...")
attn_head, layer = get_single_attn_head(model, 3, return_layer=True)
print(f"\tRetrieved attention head of type: {type(attn_head)}")
print(f"\tRetrieved layer of type: {type(layer)}")
print("Done!\n\n")
# print("Attention head: ", attn_head)
# print("\nType of attention head: ", type(attn_head))
# print("Num heads: ", attn_head.num_heads)
# print("Head dim: ", attn_head.head_dim)
# print("Hidden size: ", attn_head.hidden_size)
# print("\nQuery proj bias: ", attn_head.q_proj.bias)



# Let's see what happens when we input the `activation` tensor from the 
# layer_num of the outputs['hidden_states'] into the attention head. 
# The forward function should work fine with all the kwargs left in place. '
# NOTE: in reality, the layer pre-processes the `hidden_states` of shape [batch, seq_len, hidden_size]
print("Running the outputs['hidden_states'][layer_num] thru the attn_head and layer...")
print(f"\tShape of hidden_states: {outputs['hidden_states'][layer_num].shape}")

with torch.no_grad():
    attn_head_out, attn_weights_attn, past_kv = attn_head(outputs['hidden_states'][layer_num], 
                                                        output_attentions=True, 
                                                        debug=False) #NOTE: debug is a custom thing I added to the forward function in the transformers lib/ files.
    layer_out, attn_weights_layer = layer(outputs['hidden_states'][layer_num], output_attentions=True)
    # running model.model.norm() -- function owned by LlamaModel --
    #  on the layer_out, which is apparently what happens to all hidden layers 
    # NOTE: The model.model.norm function is a LlamaRMSNorm which aggregates 
    # over all the hidden states, AFTER they are computed. So we won't get the 
    # exact same output in a single pass, because the norm output depends on 
    # all the hidden layer values. 
    #
    # I am satisfied, though, that we are genuinely getting the right output 
    # from the attention and the layer. 
    layer_out = model.model.norm(layer_out)
print(f"\tShape of attn_head_out: {attn_head_out.shape}")
print(f"\tShape of layer_out: {layer_out.shape}")
print(f"\tShape of attn_weights_attn: {attn_weights_attn.shape}")
print(f"\tShape of attn_weights_layer: {attn_weights_layer.shape}")

# test to see if layer_out all close with outputs['hidden_states'][layer_num+1]
print("\n\nTesting if layer_out is close to outputs['hidden_states'][layer_num+1]...")
all_close = torch.allclose(layer_out, outputs['hidden_states'][layer_num+1], atol=1e-2)
print(f"\tAll close? {all_close}")
print(f"Per-token norms in layer_out: ", torch.linalg.vector_norm(layer_out, dim=-1))
print(f"Per-token norms in outputs['hidden_states'][layer_num+1]: ", 
      torch.linalg.vector_norm(outputs['hidden_states'][layer_num+1], dim=-1))
print(f"Per-token norm in attention output: ", torch.linalg.vector_norm(attn_head_out, dim=-1))
pdb.set_trace()


print("Done!\n\n")





print(f"\n\nDone with theorem validation script. Goodbye!")

