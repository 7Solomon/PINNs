import torch
import numpy as np

class FourierFeatureTransform(torch.nn.Module):
    """
    A PyTorch module for applying Random Fourier Feature mapping.
    This is a more robust alternative to a function with a static variable.
    """
    def __init__(self, in_dim, num_features=256, sigma=10.0):
        super().__init__()
        self.in_dim = in_dim
        print('in_dim', in_dim)
        self.num_features = num_features
        
        B = torch.randn((in_dim, num_features)) * sigma
        self.register_buffer('B', B)

    def forward(self, x):
        """Applies the Fourier feature mapping."""
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Input dimension mismatch. Expected {self.in_dim}, got {x.shape[-1]}")
            
        # Project the input
        x_proj = 2 * np.pi * torch.matmul(x, self.B)
        
        # Concatenate sine and cosine features
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)

#def fourier_transform(x, num_features=20, sigma=1.0):
#    #print(f"Applying Fourier transform with {num_features} features and sigma={sigma}")
#    #print(f"Input shape: {x.shape}")
#    in_dim = x.shape[-1]
#
#    if not hasattr(fourier_transform, 'B'):
#        B = torch.randn((in_dim, num_features)) * sigma
#        fourier_transform.B = B.to(x.device)  # wild shit here, saves in function
#
#    #print(f"Random projection matrix B shape: {fourier_transform.B.shape}")
#    x_proj = 2 * np.pi * torch.matmul(x, fourier_transform.B)         # [batch_size, num_features]
#    output  = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)  # [batch_size, 2 * num_features]
#    #print(output.shape)
#    return output
