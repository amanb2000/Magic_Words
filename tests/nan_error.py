from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM
import torch
import pdb

model_name = "TheBloke/Llama-2-7B-GPTQ"
model_basename = "model"
use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name,
    model_basename=model_basename,
    inject_fused_attention=False, # Required for Llama 2 70B model at this time.
    use_safetensors=True,
    trust_remote_code=False,
    device_map='auto',
    use_triton=use_triton,
    quantize_config=None)
model.eval()


device = model.device

input_ids = torch.tensor([[ 2071, 2, 5752]]).to(device)

input_ids_tokenized = tokenizer.encode(tokenizer.decode(input_ids[0]), return_tensors='pt')[None, :]
logits = model(input_ids_tokenized).logits
print(logits)