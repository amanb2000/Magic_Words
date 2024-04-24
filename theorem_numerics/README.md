# Theorem Numerics

Here we demonstrate the theorem in action by feeding a self-attention layer real
inputs $X_0$ (sampled activations from Llama-3 8b during inference) and
determining the output controllability (as defined in the Self-Attention
Controllability Theorem) using bounded control input representations. 

We demonstrate the strict unreachability of outputs $Y$ outside the reachable 
set of outputs surrounding $Y_x$ is no greater than $\beta(k, X_0)$. 


## Setup (Llama Deconstruction)
```bash
ln theorem_numerics/modeling_llama.py venv/lib/python3.9/site-packages/transformers/models/llama/modeling_llama.py
```

