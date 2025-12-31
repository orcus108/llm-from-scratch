# llm-from-scratch

(work in progress)

code written while reading
*Build a Large Language Model from Scratch* by Sebastian Raschka. 

the goal is to understand how llms actually work by building each component.

covered so far
- built bpe tokenization and token embedding layers
- implemented self-attention from first principles: qkv projections, causal masking, and multi-head attention
- implemented a full gpt-2 (124M) architecture from scratch, including layernorm, residual connections, gelu-based ffn blocks, and greedy decoding at inference time
- pre-trained the 124M model on "The Verdict" short story

### repo structure

```text
llm-from-scratch/
├── notebooks/              # step-by-step implementations and experiments
│   ├── 01_processing_text.ipynb
│   ├── 02_attention.ipynb
│   ├── 03_implementing_GPT_model.ipynb
│   └── 04_pretraining.ipynb
│
├── models/                 # (planned) reusable PyTorch model components
├── training/               # (planned) training loops and utilities
├── experiments/            # (planned) ablations and scaling experiments
│
├── .gitignore
├── LICENSE
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