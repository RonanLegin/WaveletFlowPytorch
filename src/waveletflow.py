import torch.nn as nn
from src.dwt.dwt import Dwt
from src.dwt.wavelets import Haar
from src.nf.glow import Glow
import math
import torch
import numpy as np

class WaveletFlow(nn.Module):
    def __init__(self, cf, cond_net, partial_level=-1):
        super().__init__()
        self.n_levels = cf.nLevels
        self.base_level = cf.baseLevel
        self.partial_level = partial_level
        self.wavelet = Haar()
        self.dwt = Dwt(wavelet=self.wavelet)
        self.conditioning_network = cond_net
        self.cf = cf

        if partial_level == -1 or partial_level == self.base_level:
            base_size = 2 ** self.base_level
            cf.K = cf.stepsPerResolution[partial_level]
            cf.L = cf.stepsPerResolution_L[partial_level]
            if self.cf.imShape[0] == 1:
                shape = (cf.imShape[0] * 2, 1, 1) # quick fix for split base flow
            else:
                shape = (cf.imShape[0], 1, 1) # quick fix for split base flow
            self.base_flow = Glow(cf, shape, False)
        else:
            self.base_flow = None
        
        start_flow_padding = [None] * self.base_level
        self.sub_flows = start_flow_padding + [self.base_flow]
        
        for level in range(self.base_level + 1, self.n_levels + 1):
            if partial_level != -1 and partial_level != level:
                self.sub_flows.append(None)
            else:
                h = 2**(level-1)
                w = 2**(level-1)
                cf.K = cf.stepsPerResolution[level-1]
                cf.L = cf.stepsPerResolution_L[level-1]
                shape = (cf.imShape[0] * 3, h, w)
                self.sub_flows.append(Glow(cf, shape, cf.conditional))

        self.sub_flows = nn.ModuleList(self.sub_flows)

    def forward(self, x, partial_level=-1):
        latents = []
        low_freq = x 
        for level in range(self.n_levels, self.base_level-1, -1):
            if level == partial_level or partial_level == -1:
                if level == self.base_level:
                    flow = self.base_flow
                    conditioning = None
                    low = dwt_components['low']
                    # Check if the channel dimension is 1; if so, double it
                    if self.cf.imShape[0] == 1:
                        low = low.repeat(1, 2, 1, 1)  # This repeats the channel dimension
                    res = flow.forward(low, conditioning=conditioning)
                else:
                    dwt_components = self.dwt.forward(low_freq)
                    low_freq = dwt_components['low']
                    conditioning = self.conditioning_network.encoder_list[level](low_freq)
                    flow = self.sub_flows[level]
                    res = flow.forward(dwt_components['high'], conditioning=conditioning)

                latents.append(res["latent"])
                b, c, h, w = low_freq.shape
                res["likelihood"] -= (c * h * w * torch.log(torch.tensor(0.5)) * (self.n_levels - level)) /  (math.log(2.0) * c * h * w)
                x = torch.abs(dwt_components['high'])
                if partial_level != -1:
                    break 
            
            else:
                if self.partial_level <= 8 and level > 8:
                    pass
                else:
                    dwt_components = self.dwt.forward(low_freq)
                    low_freq = dwt_components['low']
                latents.append(None)

        return {"latent":latents, "likelihood":res["likelihood"]}


    @torch.no_grad()
    def latent_to_data(self, latents):
        """
        Inverse pass from latents to data.
        """
        # Invert base level latent
        base, _ = self.base_flow.forward(z=latents[self.base_level], temperature=1.0, reverse=True)
        #base = base_data.data
        # Prepare lists
        start_padding = [None]*self.base_level
        reconstructions = start_padding + [base]
        details = start_padding + [None]
        # Reconstruct higher resolutions
        for level in range(self.base_level+1, self.n_levels+1):
            latent = latents[level]
            base = reconstructions[-1]
            super_res, level_sample = self.latent_to_super_res(latent, level, base)

            reconstructions.append(super_res)
            details.append(level_sample)

        out = {
            "reconstructions": reconstructions,
            "details": details
        }

        return out

    @torch.no_grad()
    def latent_to_super_res(self, latent, level, base):
        """
        Invert one level of super-resolution from latents.
        """
        flow = self.sub_flows[level]

        # Compute conditioning from the current base
        conditioning = self.conditioning_network.encoder_list[level](base)
        # Invert flow for the latent at this level
        if level == 1:
            base = base[:,:1]
            conditioning = conditioning[:,:1]

        level_sample, _ = flow.forward(z=latent, conditioning=conditioning, temperature=1.0, reverse=True)
        super_res = self.dwt.inverse(base, level_sample)

        return super_res, level_sample

    @torch.no_grad()
    def sample_latents(self, n_batch=1, temperature=1.0):
        """
        Sample latents from all flows based on their input shapes.
        """
        latents = [None]*self.base_level

        for flow in self.sub_flows[self.base_level:]:
            if flow is not None:
                # Access the stored input shape (C, H, W)
                _, c, h, w = flow.flow.output_shapes[-1]
                # Sample from a Normal(0,1) distribution, then scale by temperature
                latent = torch.randn(n_batch, c, h, w, device=next(flow.parameters()).device) * temperature
                latents.append(latent)
            else:
                latents.append(None)

        return latents

    
    # @torch.no_grad()
    # def latent_to_data(self, latents, partial_level=-1):
    #     """
    #     Inverse pass from latents to data:
    #     Starting from the base level latent, invert the flows and reconstruct the image level by level.

    #     Args:
    #         latents (list): A list of latent tensors for each level. 
    #                         `latents[self.base_level]` is the base level latent,
    #                         `latents[level]` are higher level latents or None if no latent at that level.
    #         partial_level (int): If not -1, only reconstruct up to partial_level.

    #     Returns:
    #         dict: Contains:
    #             "data": The reconstructed image (if partial_level == -1 or partial_level == self.n_levels)
    #             "reconstructions": A list with the reconstructed images at each level
    #             "details": A list with the detail (high-frequency) components at each level
    #             "likelihood": If computed, the sum or final likelihood after inversion
    #     """

    #     # Invert from the base level
    #     base_flow = self.sub_flows[self.base_level]
    #     base_latent = latents[self.base_level]

    #     # No conditioning at the base level
    #     base_res = base_flow.inverse(base_latent, conditioning=None)
    #     low_freq = base_res["data"]

    #     start_padding = [None]*self.base_level
    #     reconstructions = start_padding + [low_freq]
    #     details = start_padding + [None]

    #     # Initialize likelihood tracking if needed
    #     ld = base_res.get("ld", 0.0)
    #     ld_base = base_res.get("ld_base", 0.0)
    #     ldj = base_res.get("ldj", 0.0)

    #     # Move up the hierarchy to reconstruct higher levels
    #     # Note: Forward goes from n_levels down to base_level
    #     # Inverse goes from base_level up to n_levels
    #     for level in range(self.base_level+1, self.n_levels+1):
    #         if partial_level != -1 and level > partial_level:
    #             # If partial_level is specified and we reached beyond it, stop
    #             break

    #         # If no latent for this level (e.g., partial scenario), skip
    #         if latents[level] is None:
    #             # Just append None and continue (no super-resolution step)
    #             reconstructions.append(low_freq)
    #             details.append(None)
    #             continue

    #         # Get conditioning at this level
    #         conditioning = self.conditioning_network.encoder_list[level](low_freq)

    #         # Invert the flow to get high_freq
    #         flow = self.sub_flows[level]
    #         level_latent = latents[level]
    #         level_res = flow.inverse(level_latent, conditioning=conditioning)
    #         high_freq = level_res["data"]

    #         # Combine low_freq and high_freq using inverse DWT
    #         # Ensure that `self.dwt.inverse` performs the exact inverse of `self.dwt.forward`
    #         # and returns the reconstructed image and any ldj terms if applicable.
    #         # If your inverse does not produce ldj_haar, set it to zero or handle accordingly.
    #         recon, ldj_haar = self.dwt.inverse(low_freq, high_freq)

    #         # Update likelihood terms if your model uses them
    #         # In forward, you had:
    #         # res["likelihood"] -= (some term)
    #         # If you tracked ld, ld_base, ldj similarly in inverse steps, adjust here.
    #         # For now, just mimic the structure:
    #         ld += level_res.get("ld", 0.0) - ldj_haar
    #         ld_base += level_res.get("ld_base", 0.0)
    #         ldj += level_res.get("ldj", 0.0) - ldj_haar

    #         # Update low_freq for the next iteration
    #         low_freq = recon
    #         reconstructions.append(recon)
    #         details.append(high_freq)

    #     # Once done, `low_freq` is the full-resolution image
    #     out = {
    #         "data": low_freq,
    #         "reconstructions": reconstructions,
    #         "details": details,
    #         "ld": ld,
    #         "ld_base": ld_base,
    #         "ldj": ldj
    #     }

    #     return out

    # @torch.no_grad()
    # def sample(self, n_samples, device='cuda'):
    #     """
    #     Hierarchical sampling:
    #     1. Sample from the base level latent space.
    #     2. Invert the base flow to get the coarsest low_freq image.
    #     3. Iteratively sample high-frequency latents and invert flows to get high-frequency coeffs.
    #     4. Apply inverse DWT at each level to reconstruct a higher-resolution image.
    #     5. Return the final reconstructed image.
    #     """

    #     # Step 1: Sample from the base distribution
    #     flow = self.sub_flows[self.base_level]
    #     if flow is None:
    #         raise ValueError("Base flow not defined for this model configuration.")
        
    #     base_shape = (n_samples,) + flow.input_shape  # (B, C, H, W)
    #     z_base = torch.randn(base_shape, device=device)

    #     # Conditioning at base level is None
    #     conditioning = None
    #     low_freq = flow.inverse(z_base, conditioning=conditioning)  # Inverse of base flow

    #     # Step 2: Iterate through higher levels
    #     for level in range(self.base_level + 1, self.n_levels + 1):
    #         flow = self.sub_flows[level]
    #         if flow is None:
    #             # This means we are not refining at this level (partial_level scenario)
    #             continue

    #         # Get conditioning from current low_freq image
    #         conditioning = self.conditioning_network.encoder_list[level](low_freq)

    #         # Sample z for the high frequency components
    #         # shape: (B, cf.imShape[0]*3, 2^(level-1), 2^(level-1))
    #         c = self.conditioning_network.cf.imShape[0]
    #         h = 2 ** (level - 1)
    #         w = 2 ** (level - 1)
    #         z_high = torch.randn(n_samples, c * 3, h, w, device=device)
            
    #         # Inverse flow for high frequency
    #         high_freq = flow.inverse(z_high, conditioning=conditioning)

    #         # Inverse DWT to combine low_freq and high_freq into the next-scale image
    #         low_freq = self.dwt.inverse(low_freq, high_freq)

    #     # After finishing all levels, low_freq is now the full-resolution image
    #     return low_freq
    

