"""
Decode latent embeddings (z_q) into motion sequences.

This module provides functions to:
1. Quantize shifted embeddings: Q(z_e^shifted)
2. Decode quantized embeddings: D(Q(z_e^shifted)) → motion sequences
"""

import os
import sys
import torch
import numpy as np
import h5py
from typing import Dict, List, Optional, Union

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import models.vqvae as vqvae
import options.option_vq as option_vq


class LatentDecoder:
    """
    Decoder for VQ-VAE latent embeddings to motion sequences.
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 dataname: str = 't2m',
                 device: str = 'cuda',
                 **model_kwargs):
        """
        Initialize the decoder with a trained VQ-VAE model.
        
        Args:
            checkpoint_path: Path to VQ-VAE checkpoint (.pth file)
            dataname: Dataset name ('t2m' or 'kit')
            device: Device to run on ('cuda' or 'cpu')
            **model_kwargs: Additional model arguments (nb_code, code_dim, etc.)
        """
        self.dataname = dataname
        self.device = device
        
        # Get default args
        args = option_vq.get_args_parser()
        
        # Set dataname
        args.dataname = dataname
        
        # Override with provided kwargs
        for key, value in model_kwargs.items():
            setattr(args, key, value)
        
        # Set defaults if not provided
        if not hasattr(args, 'nb_code') or args.nb_code is None:
            args.nb_code = 512
        if not hasattr(args, 'code_dim') or args.code_dim is None:
            args.code_dim = 512
        if not hasattr(args, 'output_emb_width') or args.output_emb_width is None:
            args.output_emb_width = 512
        if not hasattr(args, 'down_t') or args.down_t is None:
            args.down_t = 2
        if not hasattr(args, 'stride_t') or args.stride_t is None:
            args.stride_t = 2
        if not hasattr(args, 'width') or args.width is None:
            args.width = 512
        if not hasattr(args, 'depth') or args.depth is None:
            args.depth = 3
        if not hasattr(args, 'dilation_growth_rate') or args.dilation_growth_rate is None:
            args.dilation_growth_rate = 3
        if not hasattr(args, 'vq_act') or args.vq_act is None:
            args.vq_act = 'relu'
        if not hasattr(args, 'quantizer') or args.quantizer is None:
            args.quantizer = 'ema_reset'
        
        # Create model
        self.net = vqvae.HumanVQVAE(
            args,
            args.nb_code,
            args.code_dim,
            args.output_emb_width,
            args.down_t,
            args.stride_t,
            args.width,
            args.depth,
            args.dilation_growth_rate,
            activation=args.vq_act
        )
        
        # Load checkpoint
        print(f"Loading VQ-VAE checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        self.net.load_state_dict(ckpt['net'], strict=True)
        self.net.eval()
        self.net.to(device)
        print(f"✓ Model loaded on {device}")
    
    def quantize_embeddings(self, embeddings: np.ndarray) -> torch.Tensor:
        """
        Quantize embeddings to get code indices.
        
        This implements: Q(z_e^shifted) - nearest-neighbor quantization
        
        Args:
            embeddings: Embeddings of shape (T', D) or (N, T', D)
        
        Returns:
            Code indices of shape (N, T') where N is batch size
        """
        # Convert to torch tensor
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        
        # Handle single sequence
        if len(embeddings.shape) == 2:
            embeddings = embeddings.unsqueeze(0)  # (1, T', D)
        
        N, T_prime, D = embeddings.shape
        
        # Move to device
        embeddings = embeddings.to(self.device)
        
        # Flatten for quantization: (N, T', D) -> (N*T', D)
        embeddings_flat = embeddings.contiguous().view(-1, D)
        
        # Quantize: get code indices
        # The quantizer expects (NT', D) and returns (NT',) indices
        # This applies nearest-neighbor quantization independently to each timestep
        with torch.no_grad():
            code_indices = self.net.vqvae.quantizer.quantize(embeddings_flat)
        
        # Reshape back: (N*T',) -> (N, T')
        code_indices = code_indices.view(N, -1)
        
        return code_indices
    
    def decode(self, code_indices: torch.Tensor) -> np.ndarray:
        """
        Decode code indices to motion sequences.
        
        Args:
            code_indices: Code indices of shape (N, T') or (T',)
        
        Returns:
            Motion sequences of shape (N, T, 263) or (T, 263) for t2m
        """
        # Handle single sequence
        if len(code_indices.shape) == 1:
            code_indices = code_indices.unsqueeze(0)  # (1, T')
        
        # Move to device
        code_indices = code_indices.to(self.device)
        
        # Decode
        with torch.no_grad():
            motions = self.net.forward_decoder(code_indices)  # (N, T, 263)
        
        # Convert to numpy
        motions = motions.cpu().numpy()
        
        # Remove batch dimension if single sequence
        if motions.shape[0] == 1:
            motions = motions[0]
        
        return motions
    
    def quantize_and_decode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Quantize embeddings and decode to motion sequences in one step.
        
        This implements: x̂_α = D(Q(z_e^shifted))
        
        Args:
            embeddings: Shifted embeddings of shape (T', D) or (N, T', D)
        
        Returns:
            Motion sequences of shape (T, 263) or (N, T, 263)
        """
        # Quantize
        code_indices = self.quantize_embeddings(embeddings)
        
        # Decode
        motions = self.decode(code_indices)
        
        return motions


def load_decoder(checkpoint_path: str = "pretrained/VQVAE/net_last.pth",
                 dataname: str = 't2m',
                 device: str = 'cuda',
                 **model_kwargs) -> LatentDecoder:
    """
    Load and return a LatentDecoder instance.
    
    Args:
        checkpoint_path: Path to VQ-VAE checkpoint
        dataname: Dataset name ('t2m' or 'kit')
        device: Device to run on
        **model_kwargs: Additional model arguments
    
    Returns:
        LatentDecoder instance
    """
    return LatentDecoder(checkpoint_path, dataname, device, **model_kwargs)


def decode_shifted_embeddings_from_hdf5(hdf5_path: str,
                                        decoder: LatentDecoder,
                                        output_dir: str = "clustering/outputs/decoded_motions",
                                        save_npy: bool = True) -> Dict[float, np.ndarray]:
    """
    Load shifted embeddings from HDF5 and decode them to motions.
    
    Args:
        hdf5_path: Path to HDF5 file with shifted embeddings
        decoder: LatentDecoder instance
        output_dir: Directory to save decoded motions
        save_npy: Whether to save decoded motions as .npy files
    
    Returns:
        Dictionary mapping alpha values to decoded motion sequences
    """
    print(f"Loading shifted embeddings from {hdf5_path}...")
    
    decoded_motions = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        source_id = f.attrs.get('source_sample_id', 'unknown')
        direction_desc = f.attrs.get('direction_description', 'unknown')
        
        print(f"Source sample: {source_id}")
        print(f"Direction: {direction_desc}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each alpha value
        for group_name in f.keys():
            if not group_name.startswith('alpha_'):
                continue
            
            group = f[group_name]
            alpha = group.attrs['alpha']
            shifted_emb = group['shifted_embeddings'][:]  # (T', D)
            
            print(f"\nProcessing α={alpha:.1f}...")
            print(f"  Embedding shape: {shifted_emb.shape}")
            
            # Decode
            motion = decoder.quantize_and_decode(shifted_emb)
            decoded_motions[alpha] = motion
            
            print(f"  Decoded motion shape: {motion.shape}")
            
            # Save if requested
            if save_npy:
                alpha_str = f"{alpha:.1f}".replace('-', 'neg').replace('.', 'p')
                output_path = os.path.join(
                    output_dir,
                    f"{source_id}_alpha_{alpha_str}_decoded.npy"
                )
                np.save(output_path, motion)
                print(f"  Saved to {output_path}")
    
    print(f"\n✓ Decoded {len(decoded_motions)} motions")
    return decoded_motions


def example_decode_interventions():
    """Example: Decode shifted embeddings from intervention experiments."""
    print("=" * 80)
    print("Decoding Latent Embeddings to Motions")
    print("=" * 80)
    
    # Load decoder
    checkpoint_path = "pretrained/VQVAE/net_last.pth"
    decoder = load_decoder(checkpoint_path, dataname='t2m', device='cuda')
    
    # Decode shifted embeddings
    hdf5_path = "clustering/outputs/shifted_embeddings_walk_to_run.h5"
    
    if os.path.exists(hdf5_path):
        decoded_motions = decode_shifted_embeddings_from_hdf5(
            hdf5_path,
            decoder,
            output_dir="clustering/outputs/decoded_motions"
        )
        
        print(f"\nDecoded {len(decoded_motions)} motion sequences")
        print("Alpha values:", sorted(decoded_motions.keys()))
    else:
        print(f"File not found: {hdf5_path}")
        print("Please run intervention_examples.py first to generate shifted embeddings.")


if __name__ == "__main__":
    example_decode_interventions()

