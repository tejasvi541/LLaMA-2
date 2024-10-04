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

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # As written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)
    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

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

