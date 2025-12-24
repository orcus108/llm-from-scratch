# llm-from-scratch
code written while working through  
*Build a Large Language Model from Scratch* by Sebastian Raschka. (work in progress)

the goal is to understand how llms actually work by building each component.

covered so far
- built bpe tokenization and token embedding layers
- implemented self-attention from first principles: qkv projections, causal masking, and multi-head attention
- implemented a full gpt-2 (124m) architecture from scratch, including layernorm, residual connections, gelu-based ffn blocks, and greedy decoding at inference time
