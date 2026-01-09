# llm-from-scratch

(work in progress)

building every single component of an LLM from scratch using PyTorch.

the goal is to understand how llms actually work by building each component.

covered so far
- built bpe tokenization and token embedding layers
- implemented self-attention: qkv projections, causal masking, and multi-head attention
- implemented rotary position embeddings (RoPE)
- implemented a full gpt-2 (124M) architecture from scratch, including layernorm, residual connections, gelu-based ffn blocks, and greedy decoding at inference time
- pre-trained the 124M model on "The Verdict" short story
- implemented decoding strategies like temperature scaling and top-k sampling
- loaded and saved the model and optimizer states
- loaded the pretrained gpt-2 models from OpenAI

### repo structure

```text
llm-from-scratch/
├── notebooks/              # step-by-step implementations and experiments
│   ├── 01_processing0text
│   ├── 02_attention
│   ├── 03_implementing-gpt-model
│   ├── 04_pretraining 
│   ├── 05_classification-finetuning
│   └── 06_instruction-finetuning  
│
├── models/                 # reusable model components
│   ├── __init__.py
│   └── modules.py          
│
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

### running locally

```bash
# create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# launch notebooks
jupyter notebook
```

code written while reading
*Build a Large Language Model from Scratch* by Sebastian Raschka. 