import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        ''' Initializes the SelfAttention module.
        
        Args:
            n_heads (int): The number of attention heads.
            d_embed (int): The embedding dimension.
            in_proj_bias (bool, optional): Whether to use bias in the input projection. Defaults to True.
            out_proj_bias (bool, optional): Whether to use bias in the output projection. Defaults to True.'''
            
        super().__init__()
        
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_embed = d_embed // n_heads
        
    def forward(self, x: torch.Tensor, casual_mask=False):
        # x: (Batch_Size, seq_Len, Dim)
    
        ''' Calculates the self-attention of the input tensor.
        
        Args:
            x (torch.Tensor): The input tensor of shape (Batch_Size, seq_Len, Dim).
            casual_mask (bool, optional): Whether to apply a casual mask to the attention weights. Defaults to False.
        
        Returns:
            torch.Tensor: The output tensor of shape (Batch_Size, seq_Len, Dim).
        '''
        input_shape = x.shape
        batch_Size, sequence_length, d_embed = input_shape
        
        interim_shape = (batch_Size, sequence_length, self.n_heads, self.d_embed)
        
        # (Batch_Size, seq_Len, Dim) -> (Batch_Size, seq_Len, Dim * 3) -> 3 tensors of shape (Batch_Size, seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # Batch_Size, seq_Len, Dim -> (Batch_Size, seq_Len, H, Dim / H) -> (Batch_Size, H, seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        
        # (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)
        
        if casual_mask:
            # Mask where the upper triangle (abve the principal diagonal) is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
            
        weight /= math.sqrt(self.d_head)
        
        weight = F.softmax(weight, dim=-1)
        
        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v
        
        # Batch_Size, H, seq_Len, Dim / H -> (Batch_Size, seq_Len, H, Dim / H)
        output = output.transpose(1,2)
        
        output = output.reshape(input_shape)
        
        output = self.out_proj(output)
        
        # (Batch_Size, seq_Len, Dim)
        return output
            