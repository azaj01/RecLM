# RecLM: Recommendation Instruction Tuning

<img src="./RecLM_fig.png" style="zoom:100%;" />

This is the codes and dataset for **RecLM** proposed in the paper **RecLM: Recommendation Instruction Tuning.**

## 📝 Environment

For running base recommendation models:

- python==3.9.13

- numpy==1.23.1

- torch==1.11.0

- scipy==1.9.1

For tuning LLMs:

- wandb==0.16.2 

- transformers== 4.36.2

- trl==0.7.9

- peft==0.7.2

## 📚 Datasets

| Statistics       | MIND    | Netflix | Industrial |
| ---------------- | ------- | ------- | ---------- |
| # User           | 57128   | 16835   | 117433     |
| # Overlap. Item  | 1020    | 6232    | 72417      |
| # Snapshot       | daily   | yearly  | daily      |
| **Training Set** |         |         |            |
| # Item           | 2386    | 6532    | 152069     |
| # Interactions   | 89734   | 1655395 | 858087     |
| # Sparsity       | 99.934% | 98.495% | 99.995%    |
| **Test Set**     |         |         |            |
| # Item           | 2461    | 8413    | 158155     |
| # Interactions   | 87974   | 1307051 | 876415     |
| # Sparsity       | 99.937% | 99.077% | 99.995%    |

## 🚀 How to run the codes

For running base recommendation models (e.g., BiasMF):

```python
cd ./base_models/BiasMF/
python Main.py --data {dataset}
```
