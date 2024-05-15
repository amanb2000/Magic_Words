# %%
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import torch

# %%
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

# %%
input_ids = torch.tensor([[ 2071, 2, 5752]]).to(device)

logits = model(input_ids).logits
print(logits)