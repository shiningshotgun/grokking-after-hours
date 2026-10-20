# Transformer model. Architecture copied/adapted from openai/grok (transformer.py), MIT.
# https://github.com/openai/grok (transformer.py), MIT license.
import math
import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    def __init__(self, d_model, d_key):
        super().__init__()
        self.d_key = d_key
        self.Wq = nn.Linear(d_model, d_key, bias=False)
        self.Wk = nn.Linear(d_model, d_key, bias=False)
        self.Wv = nn.Linear(d_model, d_key, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_key)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.softmax(attn)
        return torch.matmul(attn, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        d_key = d_model // heads
        self.heads = nn.ModuleList([AttentionHead(d_model, d_key) for _ in range(heads)])
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        return self.Wo(out)


class FFN(nn.Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model, bias=False),
            nn.ReLU(),
            nn.Linear(mult * d_model, d_model, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
        self.ffn_drop = nn.Dropout(p=dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        a1 = self.attn(x, mask)
        a1 = self.attn_norm(x + a1)
        a2 = self.ffn(a1)
        a2 = self.ffn_drop(a2)
        a2 = self.ffn_norm(a1 + a2)
        return a2


class Transformer(nn.Module):
    def __init__(self, n_layers=2, n_heads=4, d_model=128, dropout=0.0,
                 max_context_len=16, vocab_len=239):
        super().__init__()
        self.d_model = d_model
        self.vocab_len = vocab_len
        self.embedding = nn.Embedding(vocab_len, d_model)
        # sinusoidal position encoding (official formula)
        pe = self._position_encoding(max_context_len, d_model)
        self.register_buffer("position_encoding", pe)
        self.register_buffer("self_attn_mask", self.make_mask(max_context_len))
        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_len, bias=False)

    @staticmethod
    def make_mask(context_len):
        return torch.ones([context_len, context_len]).tril()

    @classmethod
    def _position_encoding(cls, context_len, d_model):
        rows = []
        for pos in range(context_len):
            row = []
            for i in range(d_model):
                if i % 2 == 0:
                    row.append(math.sin(pos / (10000 ** (i / d_model))))
                else:
                    row.append(math.cos(pos / (10000 ** ((i - 1) / d_model))))
            rows.append(torch.tensor(row))
        stack = torch.stack(rows, dim=1)  # d_model x context_len? check
        # official: stack = torch.stack(rows, dim=1); return stack.T  (context_len x d_model)
        return stack.T

    def embed(self, indices):
        context_len = indices.shape[-1]
        pe = self.position_encoding[:context_len, :]
        embedded = self.embedding(indices)
        return pe + embedded

    def forward(self, x):
        this_max = x.shape[-1]
        mask = self.self_attn_mask[:this_max, :this_max]
        x = self.embed(x)
        a = x
        for block in self.decoder:
            a = block(a, mask)
        return self.head(a)


if __name__ == "__main__":
    from data import build_split
    train_data, val_data, tokens = build_split()
    m = Transformer(vocab_len=len(tokens))
    # param breakdown
    blocks = sum(p.numel() for n, p in m.named_parameters() if n.startswith("decoder."))
    emb = m.embedding.weight.numel()
    head = m.head.weight.numel()
    total = sum(p.numel() for p in m.parameters())
    print("blocks", blocks)
    print("embedding", emb)
    print("head", head)
    print("total", total)
