import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Features, Height, width)
        
        """
        Forward pass through the VAE attention block
        The input x is first normalized, then attention is applied between all the pixels of the image.
        The output of the attention is added to the input, and the result is returned.
        
        Parameters:
            x (torch.Tensor): Input to the VAE attention block, shape (Batch_Size, Features, Height, width)
        
        Returns:
            torch.Tensor: Output of the VAE attention block, shape (Batch_Size, Features, Height, width)
        """
        residue = x
        
        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, width) -> (Batch_Size, Features, Height * width)
        x = x.view(n, c, h * w)
        
        # (Batch_Size, Features, Height * width) -> (Batch_Size,  Height * width, Features)
        X = x.transpose(-1, -2)
        
        # (Batch_Size,  Height * width, Features) -> (Batch_Size,  Height * width, Features)
        x = self.attention(x)
        
        # (Batch_Size,  Height * width, Features) -> (Batch_Size,  Features, Height * width )
        x = x.transpose(-1, -2)
        
        # (Batch_Size,  Features, Height * width ) -> (Batch_Size, Features, Height, width)
        x = x.view((n, c, h, w))
        
        x += residue
        
        return x


        

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        __init__ method of the VAE Residual Block
        This creates the layers for the VAE Residual Block
        The VAE Residual Block applies two convolutions with group normalization in between
        The input is either added to the output if the number of input channels is the same as the number of output channels,
        or it is converted to the output channels using a 1x1 convolution before being added to the output
        Parameters:
            in_channels (int): The number of channels of the input
            out_channels (int): The number of channels of the output
        """
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if(in_channels == out_channels):
            self.residual_layer = nn.Identity()
        else: #  Conversion layer (typically a 1x1 convolution) that adjusts the number of channels in the skip connection to match the output channels of the main path.
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, In_Channels, Height, Width)
    
        residual = x
        
        x = self.groupnorm_1(x)
        
        x = F.silu(x)
        
        x = self.conv_1(x)
        
        x = self.groupnorm_2(x)
        
        x = F.silu(x)
        
        x = self.conv_2(x)
        
        return x + self.residual_layer(residual)
    
class VAE_Decoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512, 512),
            
            VAE_AttentionBlock(512),
            
            VAE_ResidualBlock(512, 512),
            
            VAE_ResidualBlock(512, 512),
            
            VAE_ResidualBlock(512, 512),
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height, Width)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            
            nn.GroupNorm(32, 128),
            
            nn.SiLU(),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        
        """
        Forward pass through the VAE decoder
        The input x is first normalized, then it is passed through all the modules in the decoder.
        The output of the decoder is returned.
        
        Parameters:
            x (torch.Tensor): Input to the VAE decoder, shape (Batch_Size, 4, Height / 8, Width / 8)
        
        Returns:
            torch.Tensor: Output of the VAE decoder, shape (Batch_Size, 3, Height, Width)
        """
        x /= 0.18215
        
        for module in self:
            x = module(x)
            
        # x: (Batch_Size, 3, Height, Width)
        
        return x
        
        
        