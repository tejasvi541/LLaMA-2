import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

# This class defines the Model Parameters
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number of heads for the queries
    n_kv_head: Optional[int] = None # Number of heads for the keys and values
    vocab_size: int = -1 # This will be set when we load the tokenizer
    multiple_of: int = 256 # Hidden Dimension of the feedforward network
    ff_dim_multipier: Optional[int] = None # Multiplier for the feedforward network
    norm_eps: float = 1e-6 # Epsilon value for the layer normalization

    # needed for kv cache
    max_seq_len: int = 2048 # Maximum sequence length
    max_batch_size: int = 32

    device: str = None

# Main model class
class Transformer(nn.Module):
    def __init__(self, args:ModelArgs)-> None:
        super().__init__()

        assert args.vocab_size != -1, "Please pass the vocab size"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freq_complex = precompute_theta_pos_frequencies(self.args.dim//self.args.n_heads, self.args.max_seq_len*2, device=self.args.device)
    
    def forward(self, tokens:torch.Tensor, start_pos:int):
        # (B, seq_len) -> (B, seq_len, dim)
        batch_size, seq_len = tokens.size()
        assert seq_len == 1, "This model only supports autoregressive generation not training as we are using llama 2 7B weights"

        h = self.tok_embeddings(tokens) # (B, seq_len, dim)

        # retrieve the pair(m, theta) corresponding to the positions [start_pos, start_pos+seq_len]
        freq_complex = self.freq_complex[start_pos:start_pos+seq_len]

        # Consecutively apply the encoder blocks

        for layer in self.layers:
            h = layer(h, start_pos,freq_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output

