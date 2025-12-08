"""
Full Pipeline Example: Intervention → Decode → Save Motions

This script demonstrates the complete pipeline:
1. Load embeddings
2. Apply direction interventions
3. Decode shifted embeddings to motions
4. Save decoded motions for visualization
"""

import os
import numpy as np
from linear_probing import load_transformer, get_standard_alphas
from decode_latents import load_decoder


def full_pipeline_example():
    """
    Complete pipeline: intervention → quantization → decoding → saving
    """
    print("=" * 80)
    print("Full Pipeline: Direction Intervention → Decode → Save")
    print("=" * 80)
    
    # Step 1: Load transformer and apply intervention
    print("\n" + "-" * 80)
    print("Step 1: Applying Direction Intervention")
    print("-" * 80)
    
    transformer = load_transformer()
    info = transformer.get_info()
    
    # Check available verbs
    source_verb = 'walk'
    target_verb = 'run'
    
    if source_verb not in info['available_verbs'] or target_verb not in info['available_verbs']:
        print(f"Warning: Required verbs not found.")
        print(f"Available verbs: {info['available_verbs'][:10]}")
        return
    
    # Compute direction vector
    direction = transformer.compute_direction_vector(
        'verb', source_verb,
        'verb', target_verb,
        aggregate='mean'
    )
    print(f"Direction vector: {source_verb} → {target_verb}")
    print(f"Direction shape: {direction.shape}")
    
    # Get source sample
    source_samples = transformer.verb_to_samples.get(source_verb, [])
    if not source_samples:
        print(f"No samples found for '{source_verb}'")
        return
    
    source_id = source_samples[0]
    print(f"Source sample: {source_id}")
    
    # Apply interventions
    alphas = get_standard_alphas()
    interventions = transformer.apply_intervention_batch(
        source_id, direction, alphas, use_full_sequence=True
    )
    print(f"Applied {len(interventions)} interventions with α ∈ {alphas}")
    
    # Step 2: Save shifted embeddings
    print("\n" + "-" * 80)
    print("Step 2: Saving Shifted Embeddings")
    print("-" * 80)
    
    save_path = "clustering/outputs/shifted_embeddings_walk_to_run.h5"
    transformer.save_shifted_embeddings(
        interventions,
        source_id,
        f"{source_verb}_to_{target_verb}",
        save_path
    )
    
    # Step 3: Load decoder and decode
    print("\n" + "-" * 80)
    print("Step 3: Decoding Shifted Embeddings to Motions")
    print("-" * 80)
    
    checkpoint_path = "pretrained/VQVAE/net_last.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Please update checkpoint_path to point to your VQ-VAE model.")
        return
    
    decoder = load_decoder(
        checkpoint_path=checkpoint_path,
        dataname='t2m',
        device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    )
    
    # Decode all interventions
    decoded_motions = transformer.decode_shifted_embeddings(
        interventions,
        decoder=decoder
    )
    
    # Step 4: Save decoded motions
    print("\n" + "-" * 80)
    print("Step 4: Saving Decoded Motions")
    print("-" * 80)
    
    output_dir = "clustering/outputs/decoded_motions"
    os.makedirs(output_dir, exist_ok=True)
    
    for alpha, motion in decoded_motions.items():
        alpha_str = f"{alpha:.1f}".replace('-', 'neg').replace('.', 'p')
        output_path = os.path.join(
            output_dir,
            f"{source_id}_alpha_{alpha_str}_decoded.npy"
        )
        np.save(output_path, motion)
        print(f"  α={alpha:4.1f}: Saved to {output_path} (shape: {motion.shape})")
    
    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Use render_final.py to visualize the decoded motions")
    print("  2. Or load the .npy files and visualize with your preferred tool")
    print(f"  3. Decoded motions saved in: {output_dir}")


def decode_from_saved_hdf5():
    """
    Alternative: Decode from previously saved HDF5 file
    """
    print("=" * 80)
    print("Decoding from Saved HDF5 File")
    print("=" * 80)
    
    from decode_latents import decode_shifted_embeddings_from_hdf5, load_decoder
    
    hdf5_path = "clustering/outputs/shifted_embeddings_walk_to_run.h5"
    
    if not os.path.exists(hdf5_path):
        print(f"File not found: {hdf5_path}")
        print("Please run the intervention step first.")
        return
    
    checkpoint_path = "pretrained/VQVAE/net_last.pth"
    decoder = load_decoder(checkpoint_path, dataname='t2m', device='cuda')
    
    decoded_motions = decode_shifted_embeddings_from_hdf5(
        hdf5_path,
        decoder,
        output_dir="clustering/outputs/decoded_motions"
    )
    
    print(f"\n✓ Decoded {len(decoded_motions)} motions")
    return decoded_motions


if __name__ == "__main__":
    try:
        full_pipeline_example()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "=" * 80)
        print("Trying alternative: decode from saved HDF5")
        print("=" * 80)
        try:
            decode_from_saved_hdf5()
        except Exception as e2:
            print(f"Alternative also failed: {e2}")

