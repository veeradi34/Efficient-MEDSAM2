"""
Quick fix for MSD 4-channel to 3-channel conversion
"""

import torch
import torch.nn as nn

class ChannelAdapter(nn.Module):
    """Convert multi-channel medical data to 3-channel for backbone."""
    
    def __init__(self, input_channels: int = 4, output_channels: int = 3):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        if input_channels == output_channels:
            self.channel_conv = nn.Identity()
        else:
            # Learnable channel adaptation
            self.channel_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=True)
            
            # Initialize with medical imaging knowledge
            self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights based on medical imaging modalities."""
        with torch.no_grad():
            if self.input_channels == 4 and self.output_channels == 3:
                # MSD brain tumor: [FLAIR, T1w, T1gd, T2w] -> [R, G, B]
                # Use domain knowledge: FLAIR shows edema, T1gd shows enhancement, T2w shows tumor
                self.channel_conv.weight[0, 0] = 0.5  # R: FLAIR
                self.channel_conv.weight[0, 3] = 0.5  # R: + T2w
                
                self.channel_conv.weight[1, 1] = 0.7  # G: T1w  
                self.channel_conv.weight[1, 2] = 0.3  # G: + T1gd
                
                self.channel_conv.weight[2, 2] = 0.6  # B: T1gd (contrast)
                self.channel_conv.weight[2, 3] = 0.4  # B: + T2w
                
            else:
                # Default initialization for other channel combinations
                nn.init.xavier_uniform_(self.channel_conv.weight)
                
            # Small bias initialization
            if hasattr(self.channel_conv, 'bias') and self.channel_conv.bias is not None:
                nn.init.constant_(self.channel_conv.bias, 0.01)
    
    def forward(self, x):
        """
        Forward pass with channel adaptation.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Tensor with adapted channels [B, output_channels, H, W]
        """
        if x.shape[1] == self.output_channels:
            return x
        elif x.shape[1] == self.input_channels:
            return self.channel_conv(x)
        else:
            # Handle unexpected channel counts
            if x.shape[1] > self.output_channels:
                # Take first N channels
                return x[:, :self.output_channels]
            else:
                # Repeat channels to match output
                repeats = self.output_channels // x.shape[1]
                remainder = self.output_channels % x.shape[1]
                repeated = x.repeat(1, repeats, 1, 1)
                if remainder > 0:
                    repeated = torch.cat([repeated, x[:, :remainder]], dim=1)
                return repeated

if __name__ == "__main__":
    adapter = ChannelAdapter()
    test_input = torch.randn(2, 4, 512, 512)
    output = adapter(test_input)
    print(f"Input: {test_input.shape} -> Output: {output.shape}")