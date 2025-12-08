"""
Examples of Linear Transformations in VQ-VAE Latent Space

This script demonstrates various vector arithmetic operations you can perform
on the clustered embeddings.
"""

import numpy as np
from linear_probing import load_transformer


def example_1_verb_arithmetic():
    """Example: Walk + Run - Stand (analogous to word2vec style arithmetic)"""
    print("\n" + "=" * 80)
    print("Example 1: Verb Arithmetic - Walk + Run - Stand")
    print("=" * 80)
    
    transformer = load_transformer()
    
    # Get verb centroids
    walk = transformer.get_verb_centroid('walk', aggregate='mean')
    run = transformer.get_verb_centroid('run', aggregate='mean')
    stand = transformer.get_verb_centroid('stand', aggregate='mean')
    
    # Perform arithmetic: walk + run - stand
    result = transformer.subtract(
        transformer.add(walk, run),
        stand
    )
    
    print(f"Result embedding shape: {result.shape}")
    
    # Find most similar samples
    similar = transformer.find_similar(result, top_k=5)
    print("\nMost similar samples:")
    for sid, sim in similar:
        meta = transformer.metadata[sid]
        print(f"  {sid}: similarity={sim:.3f}, verb={meta.get('cluster_label', 'N/A')}, text={meta.get('text', '')[:50]}")


def example_2_interpolation():
    """Example: Interpolate between different motion types"""
    print("\n" + "=" * 80)
    print("Example 2: Motion Interpolation")
    print("=" * 80)
    
    transformer = load_transformer()
    
    # Get two verb centroids
    verb1 = transformer.get_verb_centroid('walk', aggregate='mean')
    verb2 = transformer.get_verb_centroid('run', aggregate='mean')
    
    # Interpolate at different alpha values
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for alpha in alphas:
        interpolated = transformer.interpolate(verb1, verb2, alpha)
        similar = transformer.find_similar(interpolated, top_k=1)
        
        print(f"\nAlpha={alpha:.2f} ({(1-alpha)*100:.0f}% walk, {alpha*100:.0f}% run):")
        if similar:
            sid, sim = similar[0]
            meta = transformer.metadata[sid]
            print(f"  Most similar: {sid}, similarity={sim:.3f}, verb={meta.get('cluster_label', 'N/A')}")


def example_3_cluster_operations():
    """Example: Operations on cluster centroids"""
    print("\n" + "=" * 80)
    print("Example 3: Cluster Operations")
    print("=" * 80)
    
    transformer = load_transformer()
    info = transformer.get_info()
    
    if info['available_clusters']:
        # Get centroids of first two clusters
        cluster_ids = info['available_clusters'][:2]
        
        centroids = []
        for cid in cluster_ids:
            centroid = transformer.get_cluster_centroid(cid, aggregate='mean')
            centroids.append(centroid)
            print(f"Cluster {cid}: {len(transformer.cluster_to_samples[cid])} samples, shape={centroid.shape}")
        
        # Average the clusters
        if len(centroids) == 2:
            avg_centroid = transformer.average(centroids)
            print(f"\nAverage of clusters {cluster_ids}: shape={avg_centroid.shape}")
            
            similar = transformer.find_similar(avg_centroid, top_k=3)
            print("Most similar samples:")
            for sid, sim in similar:
                meta = transformer.metadata[sid]
                print(f"  {sid}: similarity={sim:.3f}, cluster={meta.get('cluster_id', 'N/A')}")


def example_4_scaling():
    """Example: Scale embeddings to emphasize or de-emphasize features"""
    print("\n" + "=" * 80)
    print("Example 4: Scaling Operations")
    print("=" * 80)
    
    transformer = load_transformer()
    
    # Get a verb centroid
    walk = transformer.get_verb_centroid('walk', aggregate='mean')
    
    # Scale it up (emphasize)
    emphasized = transformer.scale(walk, factor=1.5)
    
    # Scale it down (de-emphasize)
    de_emphasized = transformer.scale(walk, factor=0.5)
    
    print(f"Original walk centroid: shape={walk.shape}")
    print(f"Emphasized (1.5x): shape={emphasized.shape}")
    print(f"De-emphasized (0.5x): shape={de_emphasized.shape}")
    
    # Compare similarities
    original_sim = transformer.find_similar(walk, top_k=1)
    emphasized_sim = transformer.find_similar(emphasized, top_k=1)
    
    print("\nOriginal most similar:")
    if original_sim:
        sid, sim = original_sim[0]
        print(f"  {sid}: similarity={sim:.3f}")
    
    print("Emphasized most similar:")
    if emphasized_sim:
        sid, sim = emphasized_sim[0]
        print(f"  {sid}: similarity={sim:.3f}")


def example_5_weighted_average():
    """Example: Weighted average of multiple embeddings"""
    print("\n" + "=" * 80)
    print("Example 5: Weighted Average")
    print("=" * 80)
    
    transformer = load_transformer()
    
    # Get multiple verb centroids
    verbs = ['walk', 'run', 'jump']
    embeddings = []
    weights = [0.5, 0.3, 0.2]  # Weight walk more heavily
    
    for verb in verbs:
        try:
            emb = transformer.get_verb_centroid(verb, aggregate='mean')
            embeddings.append(emb)
            print(f"Loaded {verb}: shape={emb.shape}")
        except ValueError:
            print(f"Warning: Verb '{verb}' not found, skipping")
            continue
    
    if len(embeddings) >= 2:
        # Weighted average
        weighted_avg = transformer.average(embeddings[:len(weights)], weights=weights[:len(embeddings)])
        print(f"\nWeighted average shape: {weighted_avg.shape}")
        print(f"Weights used: {weights[:len(embeddings)]}")
        
        similar = transformer.find_similar(weighted_avg, top_k=3)
        print("\nMost similar samples:")
        for sid, sim in similar:
            meta = transformer.metadata[sid]
            print(f"  {sid}: similarity={sim:.3f}, verb={meta.get('cluster_label', 'N/A')}")


def example_6_custom_operations():
    """Example: Custom vector arithmetic operations"""
    print("\n" + "=" * 80)
    print("Example 6: Custom Operations")
    print("=" * 80)
    
    transformer = load_transformer()
    
    # Get some embeddings
    walk = transformer.get_verb_centroid('walk', aggregate='mean')
    run = transformer.get_verb_centroid('run', aggregate='mean')
    
    # Custom operation: 2*walk - run (emphasize walking, subtract running)
    result = transformer.subtract(
        transformer.scale(walk, factor=2.0),
        run
    )
    
    print(f"Operation: 2*walk - run")
    print(f"Result shape: {result.shape}")
    
    similar = transformer.find_similar(result, top_k=5)
    print("\nMost similar samples:")
    for sid, sim in similar:
        meta = transformer.metadata[sid]
        print(f"  {sid}: similarity={sim:.3f}, verb={meta.get('cluster_label', 'N/A')}")


def example_7_sequence_level_operations():
    """Example: Operations on full sequences (not aggregated)"""
    print("\n" + "=" * 80)
    print("Example 7: Sequence-Level Operations")
    print("=" * 80)
    
    transformer = load_transformer()
    
    # Get full sequences (not aggregated)
    sample_ids = list(transformer.embeddings.keys())[:2]
    
    if len(sample_ids) >= 2:
        seq1 = transformer.get_embedding(sample_ids[0], aggregate=None)  # (T', D)
        seq2 = transformer.get_embedding(sample_ids[1], aggregate=None)  # (T', D)
        
        print(f"Sequence 1: {sample_ids[0]}, shape={seq1.shape}")
        print(f"Sequence 2: {sample_ids[1]}, shape={seq2.shape}")
        
        # Interpolate between sequences (requires same length)
        if seq1.shape[0] == seq2.shape[0]:
            interpolated_seq = transformer.interpolate(seq1, seq2, alpha=0.5)
            print(f"Interpolated sequence shape: {interpolated_seq.shape}")
            
            # Aggregate for similarity search
            interpolated_mean = interpolated_seq.mean(axis=0)
            similar = transformer.find_similar(interpolated_mean, top_k=3)
            print("\nMost similar samples to interpolated sequence:")
            for sid, sim in similar:
                meta = transformer.metadata[sid]
                print(f"  {sid}: similarity={sim:.3f}, verb={meta.get('cluster_label', 'N/A')}")
        else:
            print("Sequences have different lengths, cannot interpolate directly")


if __name__ == "__main__":
    print("Linear Transformations in VQ-VAE Latent Space - Examples")
    print("=" * 80)
    
    # Run examples (comment out any you don't want to run)
    try:
        example_1_verb_arithmetic()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_2_interpolation()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        example_3_cluster_operations()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    try:
        example_4_scaling()
    except Exception as e:
        print(f"Example 4 failed: {e}")
    
    try:
        example_5_weighted_average()
    except Exception as e:
        print(f"Example 5 failed: {e}")
    
    try:
        example_6_custom_operations()
    except Exception as e:
        print(f"Example 6 failed: {e}")
    
    try:
        example_7_sequence_level_operations()
    except Exception as e:
        print(f"Example 7 failed: {e}")
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)

