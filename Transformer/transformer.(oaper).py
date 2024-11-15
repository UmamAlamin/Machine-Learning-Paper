import torch 
import torch as nn
from torch.nn import functional as F
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self,d_model:int, vocab_size: int)-> None:
        super().__init()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # mengubah indeks token menjadi representasi vektor

    def forward(self, x):
        """
        hasil dari weight dari embedding layers dikalikan dengan sqrt(d_model)
        """
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len:int, dropout:float)-> None:
        super().__init__()
        self.d_model =d_model
        self.d_model = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        # vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Aplikasikan fungsi sinus  pada dimensi genap 
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        #  Aplikasikan fungsi cos pada dimensi ganjil
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # batch dimension untuk positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)


class NormalizationLayer(nn.Module):

    def __init__(self, features:int, eps:float=10**-6)->None:
        super().__init__()
        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(features)) # untuk scaling hasil normaslisasi
        self.bias = nn.Parameter(torch.zeros(features)) # untuk shift hasil normalisasi

    def forward(self, x):
        mean= x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) /( std + self.eps) + self.bias
  
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class Head(nn.Module):
    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  
        q = self.query(x)  
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Mask future positions
        wei = F.softmax(wei, dim=-1)  
        wei = self.dropout(wei)
        v = self.value(x)  
        out = wei @ v  
        return out