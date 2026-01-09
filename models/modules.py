import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import tiktoken
import numpy as np

### from: 01_processing_text.ipynb

# ============================================================================
# Dataset utilities
# ============================================================================

class GPTDatasetV1(Dataset):
    """
    GPT-style dataset that converts raw text into overlapping input–target
    token sequences for next-token prediction.
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        """
        Args:
            txt (str): Raw text corpus.
            tokenizer: Tokenizer with an `.encode()` method.
            max_length (int): Context window length.
            stride (int): Step size between consecutive windows.
        """
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        """Return number of samples in the dataset."""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Return a single (input, target) pair.

        Returns:
            Tuple[Tensor, Tensor]: (input_ids, target_ids)
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    """
    Create a PyTorch DataLoader for GPT-style language modeling.

    Args:
        txt (str): Raw text corpus.
        batch_size (int): Batch size.
        max_length (int): Context window length.
        stride (int): Sliding window stride.
        shuffle (bool): Whether to shuffle samples.
        drop_last (bool): Whether to drop the last incomplete batch.
        num_workers (int): Number of DataLoader workers.

    Returns:
        DataLoader: Configured DataLoader instance.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )



### from: 02_attention.ipynb

# ============================================================================
# Self-attention (single-head, unmasked)
# ============================================================================

class SelfAttention_v1(nn.Module):
    """
    Minimal self-attention implementation using explicit parameter matrices.
    Intended for conceptual understanding, not efficiency.
    """

    def __init__(self, d_in, d_out):
        """
        Args:
            d_in (int): Input embedding dimension.
            d_out (int): Output embedding dimension.
        """
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        """
        Compute self-attention.

        Args:
            x (Tensor): Shape (num_tokens, d_in)

        Returns:
            Tensor: Context vectors, shape (num_tokens, d_out)
        """
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    """
    Self-attention implemented using nn.Linear layers.
    Functionally equivalent to v1 but more idiomatic PyTorch.
    """

    def __init__(self, d_in, d_out):
        """
        Args:
            d_in (int): Input embedding dimension.
            d_out (int): Output embedding dimension.
        """
        super().__init__()
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        """
        Compute self-attention.

        Args:
            x (Tensor): Shape (num_tokens, d_in)

        Returns:
            Tensor: Context vectors, shape (num_tokens, d_out)
        """
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec


# ============================================================================
# Causal (masked) attention
# ============================================================================

class CausalAttention(nn.Module):
    """
    Single-head causal self-attention with a triangular mask
    to prevent attending to future tokens.
    """

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        """
        Args:
            d_in (int): Input embedding dimension.
            d_out (int): Output embedding dimension.
            context_length (int): Maximum sequence length.
            dropout (float): Dropout probability.
            qkv_bias (bool): Whether to use bias in QKV projections.
        """
        super().__init__()
        self.d_out = d_out

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x):
        """
        Compute causal self-attention.

        Args:
            x (Tensor): Shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Context vectors, shape (batch_size, num_tokens, d_out)
        """
        b, num_tokens, _ = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


# ============================================================================
# Multi-head attention
# ============================================================================

class MultiHeadAttentionWrapper(nn.Module):
    """
    Naive multi-head attention implemented by concatenating
    multiple independent causal attention heads.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        Args:
            d_in (int): Input embedding dimension.
            d_out (int): Per-head output dimension.
            context_length (int): Maximum sequence length.
            dropout (float): Dropout probability.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): Whether to use bias in QKV projections.
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        """
        Apply all attention heads and concatenate outputs.

        Args:
            x (Tensor): Shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Shape (batch_size, num_tokens, num_heads * d_out)
        """
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    """
    Efficient multi-head causal self-attention using a single
    QKV projection and batched matrix operations.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        Args:
            d_in (int): Input embedding dimension.
            d_out (int): Total output embedding dimension.
            context_length (int): Maximum sequence length.
            dropout (float): Dropout probability.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): Whether to use bias in QKV projections.
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x):
        """
        Compute multi-head causal self-attention.

        Args:
            x (Tensor): Shape (batch_size, num_tokens, d_in)

        Returns:
            Tensor: Shape (batch_size, num_tokens, d_out)
        """
        b, num_tokens, _ = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        attn_weights = torch.softmax(
            attn_scores / self.head_dim ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        return self.out_proj(context_vec)

  

### from notebook 3: 03_implementing_gpt_model.ipynb

# ============================================================================
# Normalization layers
# ============================================================================

class LayerNorm(nn.Module):
    """
    Layer Normalization module.

    Normalizes inputs across the last dimension and applies
    learnable scale and shift parameters.

    This implementation matches the behavior used in
    Transformer-based language models (e.g., GPT).
    """

    def __init__(self, emb_dim):
        """
        Args:
            emb_dim (int): Dimensionality of the embedding/features.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.eps = 1e-5

    def forward(self, x):
        """
        Apply layer normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (..., emb_dim).

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# ============================================================================
# Activation functions
# ============================================================================

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation.

    Uses the tanh-based approximation introduced in the
    original Transformer and GPT papers.
    """

    def __init__(self):
        """Initialize the GELU activation."""
        super().__init__()

    def forward(self, x):
        """
        Apply the GELU activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated tensor.
        """
        return 0.5 * x * (
            1.0
            + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )


# ============================================================================
# Feedforward networks
# ============================================================================

class FeedForward(nn.Module):
    """
    Position-wise feedforward network used inside
    a Transformer block.

    Consists of two linear layers with a GELU activation
    and an expansion factor of 4.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (dict): Model configuration dictionary containing:
                - emb_dim (int): Embedding dimensionality.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """
        Apply the feedforward network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, emb_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        return self.layers(x)


# ============================================================================
# Transformer block
# ============================================================================

class TransformerBlock(nn.Module):
    """
    Single Transformer block.

    Applies multi-head causal self-attention followed by a
    position-wise feedforward network, using pre-layer
    normalization and residual connections.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (dict): Model configuration dictionary containing:
                - emb_dim (int): Embedding dimension
                - n_heads (int): Number of attention heads
                - context_length (int): Maximum sequence length
                - drop_rate (float): Dropout probability
                - qkv_bias (bool): Whether to use bias in QKV projections
        """
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        Forward pass of the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, emb_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        # Attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Feedforward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


# ============================================================================
# Full GPT model
# ============================================================================

class GPTModel(nn.Module):
    """
    Full GPT-style Transformer language model.

    Maps a sequence of token IDs to logits over the vocabulary
    for each position. Consists of token and positional embeddings,
    a stack of Transformer blocks, a final LayerNorm, and a linear
    output head.

    This is a decoder-only Transformer architecture.
    """

    def __init__(self, cfg):
        """
        Initialize the GPT model.

        Args:
            cfg (dict): Model configuration dictionary containing:
                - vocab_size (int): Vocabulary size
                - emb_dim (int): Embedding dimension
                - context_length (int): Maximum sequence length
                - drop_rate (float): Dropout probability
                - n_layers (int): Number of Transformer blocks
        """
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        """
        Forward pass of the GPT model.

        Args:
            in_idx (torch.LongTensor): Input token indices of shape
                (batch_size, seq_len).

        Returns:
            torch.FloatTensor: Logits of shape
                (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits  


### from notebook 4: 04_pretraining.ipynb

# ============================================================================
# Tokenization utilities
# ============================================================================

def text_to_token_ids(text, tokenizer):
    """
    Convert a text string into a tensor of token IDs.

    Args:
        text (str): Input text to tokenize.
        tokenizer: Tokenizer with an `encode` method.

    Returns:
        torch.Tensor: Token IDs of shape (1, seq_len).
    """
    encoded = tokenizer.encode(
        text, allowed_special={"<|endoftext|>"}
    )
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token IDs back into a text string.

    Args:
        token_ids (torch.Tensor): Token IDs of shape (1, seq_len)
            or (seq_len,).
        tokenizer: Tokenizer with a `decode` method.

    Returns:
        str: Decoded text.
    """
    token_ids = token_ids.squeeze(0)
    return tokenizer.decode(token_ids.tolist())


# ============================================================================
# Loss computation
# ============================================================================

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Compute cross-entropy loss for a single batch.

    Args:
        input_batch (torch.Tensor): Input token IDs, shape (B, T).
        target_batch (torch.Tensor): Target token IDs, shape (B, T).
        model (nn.Module): Language model.
        device (torch.device): Computation device.

    Returns:
        torch.Tensor: Scalar loss.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Compute average loss over a DataLoader.

    Args:
        data_loader (DataLoader): Yields (input_batch, target_batch).
        model (nn.Module): Language model.
        device (torch.device): Computation device.
        num_batches (int, optional): Max batches to evaluate.

    Returns:
        float: Average loss.
    """
    if len(data_loader) == 0:
        return float("nan")

    total_loss = 0.0
    max_batches = len(data_loader) if num_batches is None else min(
        num_batches, len(data_loader)
    )

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= max_batches:
            break

        loss = calc_loss_batch(
            input_batch, target_batch, model, device
        )
        total_loss += loss.item()

    return total_loss / max_batches


# ============================================================================
# Evaluation helpers
# ============================================================================

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate model on train and validation sets.

    Args:
        model (nn.Module): Language model.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        device (torch.device): Computation device.
        eval_iter (int): Number of batches per split.

    Returns:
        tuple[float, float]: (train_loss, val_loss)
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, eval_iter
        )
    model.train()
    return train_loss, val_loss


# ============================================================================
# Text generation helpers
# ============================================================================

def generate_and_print_sample(
    model, tokenizer, device, start_context
):
    """
    Generate and print a short text sample from the model.

    Uses greedy decoding.

    Args:
        model (nn.Module): Language model.
        tokenizer: Tokenizer for encoding/decoding.
        device (torch.device): Computation device.
        start_context (str): Prompt text.
    """
    model.eval()

    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(
        start_context, tokenizer
    ).to(device)

    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )

    decoded_text = token_ids_to_text(
        token_ids, tokenizer
    )
    print(decoded_text.replace("\n", " "))

    model.train()


# ============================================================================
# Training loop
# ============================================================================

def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    """
    Train a language model with periodic evaluation and sampling.

    Args:
        model (nn.Module): Language model.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Computation device.
        num_epochs (int): Number of epochs.
        eval_freq (int): Evaluation frequency (steps).
        eval_iter (int): Batches per evaluation.
        start_context (str): Prompt for text generation.
        tokenizer: Tokenizer.

    Returns:
        tuple:
            - train_losses (list[float])
            - val_losses (list[float])
            - tokens_seen (list[int])
    """
    train_losses, val_losses, tokens_seen_track = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter,
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tokens_seen_track.append(tokens_seen)

                print(
                    f"Ep {epoch+1} | Step {global_step:06d} | "
                    f"Train {train_loss:.3f} | Val {val_loss:.3f}"
                )

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, tokens_seen_track


# ============================================================================
# Autoregressive decoding
# ============================================================================

def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None,
):
    """
    Autoregressively generate tokens from a language model.

    Supports greedy decoding, temperature sampling, and top-k filtering.

    Args:
        model (nn.Module): Language model.
        idx (torch.Tensor): Input token IDs, shape (B, T).
        max_new_tokens (int): Tokens to generate.
        context_size (int): Model context window.
        temperature (float): Sampling temperature.
        top_k (int, optional): Top-k filtering.
        eos_id (int, optional): End-of-sequence token ID.

    Returns:
        torch.Tensor: Token IDs of shape (B, T + generated).
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val.unsqueeze(-1),
                torch.tensor(-float("inf"), device=logits.device),
                logits,
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(
                logits, dim=-1, keepdim=True
            )

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# ============================================================================
# Pretrained weight loading
# ============================================================================

def assign(left, right):
    """
    Assign pretrained weights with strict shape checking.

    Args:
        left (torch.Tensor): Target tensor.
        right (array-like): Source values.

    Returns:
        nn.Parameter: New parameter with assigned values.

    Raises:
        ValueError: If shapes do not match.
    """
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch: {left.shape} vs {right.shape}"
        )

    return torch.nn.Parameter(
        torch.tensor(right, dtype=left.dtype)
    )


def load_weights_into_gpt(gpt, params):
    """
    Load GPT-2 style pretrained weights into a custom GPT model.

    Args:
        gpt (nn.Module): GPT model instance.
        params (dict): Pretrained parameter dictionary.

    Notes:
        - QKV weights are split explicitly.
        - Linear weights are transposed to PyTorch format.
        - LayerNorm parameters are assigned directly.
    """
    gpt.pos_emb.weight = assign(
        gpt.pos_emb.weight, params["wpe"]
    )
    gpt.tok_emb.weight = assign(
        gpt.tok_emb.weight, params["wte"]
    )

    for b, block in enumerate(params["blocks"]):
        q_w, k_w, v_w = np.split(
            block["attn"]["c_attn"]["w"], 3, axis=-1
        )
        q_b, k_b, v_b = np.split(
            block["attn"]["c_attn"]["b"], 3, axis=-1
        )

        att = gpt.trf_blocks[b].att

        att.W_query.weight = assign(att.W_query.weight, q_w.T)
        att.W_key.weight = assign(att.W_key.weight, k_w.T)
        att.W_value.weight = assign(att.W_value.weight, v_w.T)

        att.W_query.bias = assign(att.W_query.bias, q_b)
        att.W_key.bias = assign(att.W_key.bias, k_b)
        att.W_value.bias = assign(att.W_value.bias, v_b)

        att.out_proj.weight = assign(
            att.out_proj.weight,
            block["attn"]["c_proj"]["w"].T,
        )
        att.out_proj.bias = assign(
            att.out_proj.bias,
            block["attn"]["c_proj"]["b"],
        )

        ff = gpt.trf_blocks[b].ff.layers
        ff[0].weight = assign(
            ff[0].weight, block["mlp"]["c_fc"]["w"].T
        )
        ff[0].bias = assign(
            ff[0].bias, block["mlp"]["c_fc"]["b"]
        )
        ff[2].weight = assign(
            ff[2].weight, block["mlp"]["c_proj"]["w"].T
        )
        ff[2].bias = assign(
            ff[2].bias, block["mlp"]["c_proj"]["b"]
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            block["ln_1"]["g"],
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            block["ln_1"]["b"],
        )

        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            block["ln_2"]["g"],
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            block["ln_2"]["b"],
        )

    gpt.final_norm.scale = assign(
        gpt.final_norm.scale, params["g"]
    )
    gpt.final_norm.shift = assign(
        gpt.final_norm.shift, params["b"]
    )
    gpt.out_head.weight = assign(
        gpt.out_head.weight, params["wte"]
    )