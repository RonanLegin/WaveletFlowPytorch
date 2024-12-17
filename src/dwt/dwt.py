from importlib import machinery
import mailcap
from operator import inv
import torch
import torch.nn as nn

class Dwt(nn.Module):
    def __init__(self, wavelet):
        super().__init__()
        self.wavelet = wavelet
        self.kernel = None
        self.inv_kernel = None
        self.f = self.wavelet.factor
        self.m = self.wavelet.multiplier

    def forward(self, x):
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        H_w = self.wavelet.h
        W_w = self.wavelet.w
        high_freq = []
        low_freq = []

        assert H % H_w == 0 and W % W_w == 0, '({},{}) not dividing by {} nicely'.format(H, W, self.f)
        forward_kernel = self.make_forward_kernel(C).to(x.device)
        y = nn.functional.conv2d(x, forward_kernel, None, (2,2), 'valid')
        for i in range(C):
            low_freq.append(y[:, i*self.m:i*self.m+1, : ,:])
            high_freq.append(y[:, i*self.m+1:i*self.m+self.m, : ,:])
        
        high_freq = torch.cat(high_freq, dim=1)
        low_freq = torch.cat(low_freq, dim=1)

        components = {"low": low_freq, "high": high_freq}

        return components

    def make_forward_kernel(self, C):
        if self.kernel is not None:
            return self.kernel
        
        H_w = self.wavelet.h
        W_w = self.wavelet.w
        k = self.wavelet.kernels

        kernel = torch.zeros((C*self.m, C, H_w, W_w))
        
        for i in range(C):
            for j in range(self.m):
                kernel[i*self.m+j, i, :, :] = torch.tensor(k[j])
        
        self.kernel = kernel
        return kernel

    def make_inverse_kernel(self, C):
        # If already computed, just return it
        if self.kernel is not None:
            return self.kernel
        
        H_w = self.wavelet.h
        W_w = self.wavelet.w
        k = self.wavelet.kernels  # List of length m: [low_kernel, high_kernel1, high_kernel2, ...]

        # For the inverse:
        # The forward kernel was (C*m, C, H_w, W_w)
        # For the inverse, we need (C, C*m, H_w, W_w)
        # Essentially, we swap the role of in/out channels compared to forward.
        kernel = torch.zeros((C, C*self.m, H_w, W_w))
        
        for i in range(C):
            for j in range(self.m):
                # In the forward transform, the kernel mapping channel i to sub-band j was stored at [i*m+j, i, :, :].
                # For the inverse, we reverse this relationship:
                # inverse kernel at [i, i*m+j, :, :] = k[j]
                kernel[i, i*self.m+j, :, :] = torch.tensor(k[j], dtype=torch.float32)

        self.kernel = kernel
        return self.kernel

    def inverse(self, low, high):
        """
        low:  [B, C,   H/2, W/2]
        high: [B, C*(m-1), H/2, W/2]

        Returns:
        x: [B, C, H, W], the reconstructed image/features.
        """
        B, C, H2, W2 = low.shape
        # Combine low and high frequency components into a single tensor
        # [B, C + C*(m-1), H/2, W/2] = [B, C*m, H/2, W/2]
        x_cat = torch.cat([low, high], dim=1)

        # Compute inverse kernel and apply a transposed convolution
        inv_kernel = self.make_inverse_kernel(C).to(x_cat.device).permute(1, 0, 2, 3)
        # stride=2 will "undo" the downsampling done in the forward pass
        x = nn.functional.conv_transpose2d(x_cat, inv_kernel, stride=2)

        return x
        

# class Iwt(nn.Module):
#     def __init__(self, wavelet):
#         super().__init__()
#         self.wavelet = wavelet
#         self.f = wavelet.factor
#         self.m = wavelet.multiplier
#         self.kernel = None

#     def make_inverse_kernel(self, C):
#         # If already computed, just return it
#         if self.kernel is not None:
#             return self.kernel
        
#         H_w = self.wavelet.h
#         W_w = self.wavelet.w
#         k = self.wavelet.kernels  # List of length m: [low_kernel, high_kernel1, high_kernel2, ...]

#         # For the inverse:
#         # The forward kernel was (C*m, C, H_w, W_w)
#         # For the inverse, we need (C, C*m, H_w, W_w)
#         # Essentially, we swap the role of in/out channels compared to forward.
#         kernel = torch.zeros((C, C*self.m, H_w, W_w))
        
#         for i in range(C):
#             for j in range(self.m):
#                 # In the forward transform, the kernel mapping channel i to sub-band j was stored at [i*m+j, i, :, :].
#                 # For the inverse, we reverse this relationship:
#                 # inverse kernel at [i, i*m+j, :, :] = k[j]
#                 kernel[i, i*self.m+j, :, :] = torch.tensor(k[j], dtype=torch.float32)

#         self.kernel = kernel
#         return self.kernel

#     def forward(self, low, high):
#         """
#         low:  [B, C,   H/2, W/2]
#         high: [B, C*(m-1), H/2, W/2]

#         Returns:
#         x: [B, C, H, W], the reconstructed image/features.
#         """
#         B, C, H2, W2 = low.shape
#         # Combine low and high frequency components into a single tensor
#         # [B, C + C*(m-1), H/2, W/2] = [B, C*m, H/2, W/2]
#         x_cat = torch.cat([low, high], dim=1)

#         # Compute inverse kernel and apply a transposed convolution
#         inv_kernel = self.make_inverse_kernel(C).to(x_cat.device)
#         # stride=2 will "undo" the downsampling done in the forward pass
#         x = nn.functional.conv_transpose2d(x_cat, inv_kernel, stride=2)

#         return x