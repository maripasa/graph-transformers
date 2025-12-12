import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Para grafos, esta parte será substituída por RWPE, Laplacian PE etc.
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttentionBlock(nn.Module):
    """
    ESTA É A PARTE QUE TERÁ QUE SER ADAPTADA PARA GRAFOS.
    DOT-PRODUCT ATENÇÃO NÃO FUNCIONA EM GRAFOS SEM MÁSCARA DE ADJACÊNCIA.
    """
    def __init__(self, d_model:int, h:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.size(-1)

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask):
        B, L, _ = q.size()

        query = self.w_q(q)  # [B,L,d_model]
        key   = self.w_k(k)
        value = self.w_v(v)

        # reshape em multi-heads
        query = query.reshape(B, L, self.h, self.d_k).transpose(1, 2)
        key   = key.reshape(B, L, self.h, self.d_k).transpose(1, 2)
        value = value.reshape(B, L, self.h, self.d_k).transpose(1, 2)

        x, attn = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).reshape(B, L, self.h * self.d_k)

        return self.w_o(x)



# Layer and Norm:
# nn.LayerNorm(d_model)

# Feed Forward Network: 
# nn.Sequential(
#     nn.Linear(d_model, d_ff),
#     nn.ReLU(),
#     nn.Dropout(dropout),
#     nn.Linear(d_ff, d_model)
# )

# Encoder & Decoder:
# nn.TransformerEncoderLayer e nn.TransformerEncoder

