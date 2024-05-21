# Datasets

## Squad data loading
```bash
# from control.datapick.ai/datasets
python3 squad_loader.py

# if not already from git: 
shuf -n 100 squad_train_v2.0.jsonl > 100_squad_train_v2.0.jsonl
shuf -n 100 squad_train_v2.0.jsonl > 100_squad_validate_v2.0.jsonl
shuf -n 100 squad_train_v2.0.jsonl > 100_squad_test_v2.0.jsonl
```
