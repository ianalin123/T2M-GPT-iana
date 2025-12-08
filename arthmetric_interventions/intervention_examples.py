"""
Examples of Direction Interventions in VQ-VAE Latent Space

This script demonstrates single and compound direction interventions as described in the paper.
"""

import numpy as np
from linear_probing import load_transformer, get_standard_alphas


def example_single_direction_intervention():
    """
    Example: Single Direction Intervention
    
    Given a source motion x_0 with temporal latent codes z_e(x_0) ∈ R^{T' × 512},
    apply the same shift to all temporal positions:
    
    z_e^shifted(x_0)_t = z_e(x_0)_t + α · v_{g1→g2}  ∀ t ∈ {1, ..., T'}
    
    Test with α ∈ {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
    """
    print("\n" + "=" * 80)
    print("Single Direction Intervention Example")
    print("=" * 80)
    
    # Load transformer
    transformer = load_transformer()
    info = transformer.get_info()
    
    # Check if required verbs exist
    source_verb = 'walk'
    target_verb = 'run'
    
    if source_verb not in info['available_verbs'] or target_verb not in info['available_verbs']:
        print(f"Warning: Required verbs not found. Available: {info['available_verbs'][:10]}")
        return
    
    print(f"\nSource verb: {source_verb}")
    print(f"Target verb: {target_verb}")
    
    # Step 1: Compute direction vector v_{walk→run}
    print("\nStep 1: Computing direction vector v_{walk→run}...")
    direction = transformer.compute_direction_vector(
        'verb', source_verb,
        'verb', target_verb,
        aggregate='mean'
    )
    print(f"  Direction vector shape: {direction.shape}")
    print(f"  Direction vector norm: {np.linalg.norm(direction):.4f}")
    
    # Step 2: Get a source motion sample
    print("\nStep 2: Selecting source motion sample...")
    source_samples = transformer.verb_to_samples.get(source_verb, [])
    if not source_samples:
        print(f"  No samples found for verb '{source_verb}'")
        return
    
    source_id = source_samples[0]
    source_emb = transformer.get_embedding(source_id, aggregate=None)  # (T', D)
    print(f"  Source sample ID: {source_id}")
    print(f"  Source embedding shape: {source_emb.shape}")
    print(f"  Source metadata: {transformer.metadata[source_id].get('text', 'N/A')[:60]}")
    
    # Step 3: Apply interventions with different alpha values
    print("\nStep 3: Applying interventions with different α values...")
    alphas = get_standard_alphas()
    interventions = transformer.apply_intervention_batch(
        source_id, direction, alphas, use_full_sequence=True
    )
    
    print(f"\n  Applied {len(interventions)} interventions:")
    for alpha in alphas:
        shifted_emb = interventions[alpha]
        print(f"    α={alpha:4.1f}: shape={shifted_emb.shape}, "
              f"norm={np.linalg.norm(shifted_emb):.2f}")
    
    # Step 4: Save shifted embeddings for later quantization and decoding
    print("\nStep 4: Saving shifted embeddings...")
    save_path = "clustering/outputs/shifted_embeddings_walk_to_run.h5"
    transformer.save_shifted_embeddings(
        interventions,
        source_id,
        f"{source_verb}_to_{target_verb}",
        save_path
    )
    
    print("\n" + "-" * 80)
    print("Next steps:")
    print("  1. Load shifted embeddings from HDF5 file")
    print("  2. Apply quantization: Q(z_e^shifted)")
    print("  3. Decode: D(Q(z_e^shifted)) to get motion sequences")
    print("-" * 80)


def example_compound_direction_intervention():
    """
    Example: Compound Direction Intervention
    
    To test compositionality, combine multiple direction vectors:
    
    v_compound = Σ_i w_i · v_{g_i^(1)→g_i^(2)}
    
    Example: "running-and-jumping" motion
    v_compound = v_{walk→run} + v_{walk→jump}
    """
    print("\n" + "=" * 80)
    print("Compound Direction Intervention Example")
    print("=" * 80)
    
    # Load transformer
    transformer = load_transformer()
    info = transformer.get_info()
    
    # Check if required verbs exist
    base_verb = 'walk'
    target_verbs = ['run', 'jump']
    
    if base_verb not in info['available_verbs']:
        print(f"Warning: Base verb '{base_verb}' not found")
        return
    
    missing_verbs = [v for v in target_verbs if v not in info['available_verbs']]
    if missing_verbs:
        print(f"Warning: Target verbs not found: {missing_verbs}")
        print(f"Available verbs: {info['available_verbs'][:10]}")
        return
    
    print(f"\nBase verb: {base_verb}")
    print(f"Target verbs: {target_verbs}")
    print(f"Goal: Create 'running-and-jumping' motion")
    
    # Step 1: Compute compound direction
    print("\nStep 1: Computing compound direction vector...")
    print(f"  v_compound = v_{{{base_verb}→{target_verbs[0]}}} + v_{{{base_verb}→{target_verbs[1]}}}")
    
    compound_direction = transformer.compute_compound_direction([
        ('verb', base_verb, 'verb', target_verbs[0]),
        ('verb', base_verb, 'verb', target_verbs[1])
    ], weights=[1.0, 1.0], aggregate='mean')
    
    print(f"  Compound direction shape: {compound_direction.shape}")
    print(f"  Compound direction norm: {np.linalg.norm(compound_direction):.4f}")
    
    # Step 2: Get a source motion sample
    print("\nStep 2: Selecting source motion sample...")
    source_samples = transformer.verb_to_samples.get(base_verb, [])
    if not source_samples:
        print(f"  No samples found for verb '{base_verb}'")
        return
    
    source_id = source_samples[0]
    source_emb = transformer.get_embedding(source_id, aggregate=None)  # (T', D)
    print(f"  Source sample ID: {source_id}")
    print(f"  Source embedding shape: {source_emb.shape}")
    print(f"  Source metadata: {transformer.metadata[source_id].get('text', 'N/A')[:60]}")
    
    # Step 3: Apply interventions with different alpha values
    print("\nStep 3: Applying compound interventions with different α values...")
    alphas = get_standard_alphas()
    interventions = transformer.apply_intervention_batch(
        source_id, compound_direction, alphas, use_full_sequence=True
    )
    
    print(f"\n  Applied {len(interventions)} interventions:")
    for alpha in alphas:
        shifted_emb = interventions[alpha]
        print(f"    α={alpha:4.1f}: shape={shifted_emb.shape}, "
              f"norm={np.linalg.norm(shifted_emb):.2f}")
    
    # Step 4: Save shifted embeddings
    print("\nStep 4: Saving shifted embeddings...")
    save_path = "clustering/outputs/shifted_embeddings_compound_running_jumping.h5"
    transformer.save_shifted_embeddings(
        interventions,
        source_id,
        f"compound_{base_verb}_to_{'+'.join(target_verbs)}",
        save_path
    )
    
    print("\n" + "-" * 80)
    print("Next steps:")
    print("  1. Load shifted embeddings from HDF5 file")
    print("  2. Apply quantization: Q(z_e^shifted)")
    print("  3. Decode: D(Q(z_e^shifted)) to get motion sequences")
    print("-" * 80)


def example_custom_compound_direction():
    """
    Example: Custom compound direction with different weights
    """
    print("\n" + "=" * 80)
    print("Custom Compound Direction Example")
    print("=" * 80)
    
    transformer = load_transformer()
    info = transformer.get_info()
    
    # Example: Emphasize running more than jumping
    base_verb = 'walk'
    directions = [
        ('verb', 'walk', 'verb', 'run'),
        ('verb', 'walk', 'verb', 'jump')
    ]
    weights = [0.7, 0.3]  # Emphasize running
    
    print(f"\nBase verb: {base_verb}")
    print(f"Directions: walk→run (weight=0.7), walk→jump (weight=0.3)")
    
    if not all(v in info['available_verbs'] for v in [base_verb, 'run', 'jump']):
        print("Warning: Required verbs not found")
        return
    
    compound_direction = transformer.compute_compound_direction(
        directions, weights=weights, aggregate='mean'
    )
    
    print(f"Compound direction shape: {compound_direction.shape}")
    print(f"Compound direction norm: {np.linalg.norm(compound_direction):.4f}")
    
    # Apply to a source sample
    source_samples = transformer.verb_to_samples.get(base_verb, [])
    if source_samples:
        source_id = source_samples[0]
        shifted = transformer.apply_intervention(
            source_id, compound_direction, alpha=1.0, use_full_sequence=True
        )
        print(f"\nApplied intervention (α=1.0) to sample {source_id}")
        print(f"Shifted embedding shape: {shifted.shape}")


if __name__ == "__main__":
    print("Direction Interventions in VQ-VAE Latent Space")
    print("=" * 80)
    
    try:
        example_single_direction_intervention()
    except Exception as e:
        print(f"\nSingle direction intervention example failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        example_compound_direction_intervention()
    except Exception as e:
        print(f"\nCompound direction intervention example failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        example_custom_compound_direction()
    except Exception as e:
        print(f"\nCustom compound direction example failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)

