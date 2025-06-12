import torch
import numpy as np

def fourier_transform(x, num_features=20, sigma=1.0):
    #print(f"Applying Fourier transform with {num_features} features and sigma={sigma}")
    #print(f"Input shape: {x.shape}")
    in_dim = x.shape[-1]

    if not hasattr(fourier_transform, 'B'):
        B = torch.randn((in_dim, num_features)) * sigma
        fourier_transform.B = B.to(x.device)  # wild shit here, saves in function

    #print(f"Random projection matrix B shape: {fourier_transform.B.shape}")
    x_proj = 2 * np.pi * torch.matmul(x, fourier_transform.B)         # [batch_size, num_features]
    output  = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)  # [batch_size, 2 * num_features]
    #print(output.shape)
    return output
