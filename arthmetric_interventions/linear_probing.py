"""
Linear Transformations in VQ-VAE Latent Space

This script provides utilities for performing vector arithmetic operations
on clustered embeddings from VQ-VAE (z_qs).

Key Features:
1. Single Direction Intervention:
   Apply shift to all temporal positions: z_e^shifted(x_0)_t = z_e(x_0)_t + α · v_{g1→g2}
   
2. Compound Direction Intervention:
   Combine multiple directions: v_compound = Σ_i w_i · v_{g_i^(1)→g_i^(2)}
   
3. Standard alpha values: [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
   
The shifted embeddings can be quantized and decoded:
   x̂_α = D(Q(z_e^shifted(x_0)))
"""

import os
import h5py
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc=""):
        return iterable


class LatentSpaceTransformer:
    """
    Class for performing linear transformations in the VQ-VAE latent space.
    """
    
    def __init__(self, hdf5_path: str, use_quantized: bool = False):
        """
        Initialize the transformer with embeddings from HDF5 file.
        
        Args:
            hdf5_path: Path to the clustered HDF5 embeddings file
            use_quantized: If True, use quantized_embeddings; if False, use encoder_embeddings
        """
        self.hdf5_path = hdf5_path
        self.use_quantized = use_quantized
        self.embeddings = {}  # {sample_id: (T', D) array}
        self.metadata = {}  # {sample_id: {cluster_id, cluster_label, verb, text, ...}}
        self.cluster_to_samples = defaultdict(list)  # {cluster_id: [sample_ids]}
        self.verb_to_samples = defaultdict(list)  # {verb: [sample_ids]}
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load embeddings and metadata from HDF5 file."""
        print(f"Loading embeddings from {self.hdf5_path}...")
        
        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(
                f"Embeddings file not found: {self.hdf5_path}\n"
                f"Please run the clustering script first to generate this file."
            )
        
        with h5py.File(self.hdf5_path, 'r') as f:
            # Load root attributes
            self.n_samples = f.attrs.get('n_samples', len(list(f.keys())))
            self.embedding_dim = f.attrs.get('embedding_dim', None)
            self.n_clusters = f.attrs.get('n_clusters', None)
            
            # Load each sample
            for sample_id in tqdm(f.keys(), desc="Loading samples"):
                sample_group = f[sample_id]
                
                # Get embeddings (prefer encoder_embeddings, fallback to quantized)
                if 'encoder_embeddings' in sample_group:
                    embeddings = sample_group['encoder_embeddings'][:]  # (T', D)
                elif 'quantized_embeddings' in sample_group:
                    embeddings = sample_group['quantized_embeddings'][:]  # (T', D)
                else:
                    print(f"Warning: No embeddings found for {sample_id}")
                    continue
                
                self.embeddings[sample_id] = embeddings
                
                # Load metadata
                metadata = {
                    'cluster_id': sample_group.attrs.get('cluster_id', None),
                    'cluster_label': sample_group.attrs.get('cluster_label', None),
                    'compound_verb': sample_group.attrs.get('compound_verb', ''),
                    'text': sample_group.attrs.get('text', ''),
                    'length': sample_group.attrs.get('length', embeddings.shape[0])
                }
                self.metadata[sample_id] = metadata
                
                # Index by cluster
                if metadata['cluster_id'] is not None:
                    self.cluster_to_samples[metadata['cluster_id']].append(sample_id)
                
                # Index by verb
                if metadata['cluster_label']:
                    self.verb_to_samples[metadata['cluster_label']].append(sample_id)
                if metadata['compound_verb']:
                    verbs = metadata['compound_verb'].split('-')
                    for verb in verbs:
                        self.verb_to_samples[verb].append(sample_id)
        
        print(f"Loaded {len(self.embeddings)} samples")
        print(f"Embedding shape: {list(self.embeddings.values())[0].shape}")
        print(f"Number of clusters: {self.n_clusters}")
    
    def get_embedding(self, sample_id: str, aggregate: str = 'mean') -> np.ndarray:
        """
        Get embedding for a sample, optionally aggregated.
        
        Args:
            sample_id: Sample identifier
            aggregate: 'mean', 'max', 'flatten', or None (returns full sequence)
        
        Returns:
            Embedding vector (D,) or (T', D) if aggregate is None
        """
        if sample_id not in self.embeddings:
            raise ValueError(f"Sample {sample_id} not found")
        
        emb = self.embeddings[sample_id]  # (T', D)
        
        if aggregate is None:
            return emb
        elif aggregate == 'mean':
            return emb.mean(axis=0)  # (D,)
        elif aggregate == 'max':
            return emb.max(axis=0)  # (D,)
        elif aggregate == 'flatten':
            return emb.flatten()  # (T' * D,)
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")
    
    def get_cluster_centroid(self, cluster_id: int, aggregate: str = 'mean') -> np.ndarray:
        """
        Get the centroid (mean) of all embeddings in a cluster.
        
        Args:
            cluster_id: Cluster identifier
            aggregate: How to aggregate each sample ('mean', 'max', 'flatten')
        
        Returns:
            Centroid vector (D,) or (T' * D,) if aggregate='flatten'
        """
        sample_ids = self.cluster_to_samples.get(cluster_id, [])
        if not sample_ids:
            raise ValueError(f"Cluster {cluster_id} not found or empty")
        
        embeddings = [self.get_embedding(sid, aggregate) for sid in sample_ids]
        return np.mean(embeddings, axis=0)
    
    def get_verb_centroid(self, verb: str, aggregate: str = 'mean') -> np.ndarray:
        """
        Get the centroid of all embeddings with a given verb label.
        
        Args:
            verb: Verb label (e.g., 'walk', 'run')
            aggregate: How to aggregate each sample
        
        Returns:
            Centroid vector
        """
        sample_ids = self.verb_to_samples.get(verb, [])
        if not sample_ids:
            raise ValueError(f"No samples found for verb: {verb}")
        
        embeddings = [self.get_embedding(sid, aggregate) for sid in sample_ids]
        return np.mean(embeddings, axis=0)
    
    def add(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Add two embeddings (vector addition)."""
        return emb1 + emb2
    
    def subtract(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Subtract two embeddings (vector subtraction)."""
        return emb1 - emb2
    
    def scale(self, emb: np.ndarray, factor: float) -> np.ndarray:
        """Scale an embedding by a factor."""
        return emb * factor
    
    def interpolate(self, emb1: np.ndarray, emb2: np.ndarray, alpha: float) -> np.ndarray:
        """
        Linear interpolation between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            alpha: Interpolation factor (0.0 = emb1, 1.0 = emb2)
        
        Returns:
            Interpolated embedding
        """
        return (1 - alpha) * emb1 + alpha * emb2
    
    def average(self, embeddings: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Average multiple embeddings.
        
        Args:
            embeddings: List of embedding vectors
            weights: Optional weights for weighted average
        
        Returns:
            Averaged embedding
        """
        embeddings = np.array(embeddings)
        if weights is None:
            return embeddings.mean(axis=0)
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            return np.average(embeddings, axis=0, weights=weights)
    
    def vector_arithmetic(self, 
                         operation: str,
                         sample_ids: Optional[List[str]] = None,
                         cluster_ids: Optional[List[int]] = None,
                         verbs: Optional[List[str]] = None,
                         aggregate: str = 'mean',
                         **kwargs) -> np.ndarray:
        """
        Perform vector arithmetic operations.
        
        Examples:
            # Walk + Run - Stand
            result = transformer.vector_arithmetic(
                "walk + run - stand",
                verbs=['walk', 'run', 'stand'],
                aggregate='mean'
            )
            
            # Interpolate between two clusters
            result = transformer.vector_arithmetic(
                "cluster_0 * 0.7 + cluster_1 * 0.3",
                cluster_ids=[0, 1],
                aggregate='mean'
            )
        
        Args:
            operation: String describing the operation (e.g., "walk + run - stand")
            sample_ids: Optional list of specific sample IDs to use
            cluster_ids: Optional list of cluster IDs to use
            verbs: Optional list of verb labels to use
            aggregate: How to aggregate each sample
            **kwargs: Additional parameters (e.g., alpha for interpolation)
        
        Returns:
            Result embedding
        """
        # Parse operation and get embeddings
        # This is a simplified parser - you can extend it for more complex operations
        
        # For now, let's provide a more direct API
        raise NotImplementedError(
            "Use the individual methods (add, subtract, interpolate, etc.) "
            "or implement a parser for operation strings."
        )
    
    def find_similar(self, query_emb: np.ndarray, 
                    top_k: int = 5,
                    sample_ids: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Find most similar embeddings to a query embedding using cosine similarity.
        
        Args:
            query_emb: Query embedding vector
            top_k: Number of similar samples to return
            sample_ids: Optional list of sample IDs to search within
        
        Returns:
            List of (sample_id, similarity_score) tuples
        """
        if sample_ids is None:
            sample_ids = list(self.embeddings.keys())
        
        # Normalize query
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        
        similarities = []
        for sid in sample_ids:
            emb = self.get_embedding(sid, aggregate='mean')
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            similarity = np.dot(query_norm, emb_norm)
            similarities.append((sid, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_group_centroid(self, group_type: str, group_id: Union[int, str], 
                           aggregate: str = 'mean') -> np.ndarray:
        """
        Get centroid for a group (cluster or verb).
        
        Args:
            group_type: 'cluster' or 'verb'
            group_id: Cluster ID (int) or verb label (str)
            aggregate: How to aggregate each sample ('mean', 'max', 'flatten')
        
        Returns:
            Centroid vector (D,) or (T' * D,) if aggregate='flatten'
        """
        if group_type == 'cluster':
            return self.get_cluster_centroid(group_id, aggregate=aggregate)
        elif group_type == 'verb':
            return self.get_verb_centroid(group_id, aggregate=aggregate)
        else:
            raise ValueError(f"Unknown group_type: {group_type}. Use 'cluster' or 'verb'")
    
    def compute_direction_vector(self, 
                                source_group_type: str, source_group_id: Union[int, str],
                                target_group_type: str, target_group_id: Union[int, str],
                                aggregate: str = 'mean') -> np.ndarray:
        """
        Compute direction vector from source group to target group.
        
        Direction vector: v_{g1→g2} = centroid(g2) - centroid(g1)
        
        Args:
            source_group_type: 'cluster' or 'verb'
            source_group_id: Source cluster ID (int) or verb label (str)
            target_group_type: 'cluster' or 'verb'
            target_group_id: Target cluster ID (int) or verb label (str)
            aggregate: How to aggregate each sample ('mean', 'max', 'flatten')
        
        Returns:
            Direction vector (D,) or (T' * D,) if aggregate='flatten'
        """
        source_centroid = self.get_group_centroid(source_group_type, source_group_id, aggregate)
        target_centroid = self.get_group_centroid(target_group_type, target_group_id, aggregate)
        
        direction = target_centroid - source_centroid
        return direction
    
    def compute_compound_direction(self,
                                   directions: List[Tuple[str, Union[int, str], str, Union[int, str]]],
                                   weights: Optional[List[float]] = None,
                                   aggregate: str = 'mean') -> np.ndarray:
        """
        Compute compound direction vector by combining multiple direction vectors.
        
        v_compound = Σ_i w_i · v_{g_i^(1)→g_i^(2)}
        
        Args:
            directions: List of (source_type, source_id, target_type, target_id) tuples
                       e.g., [('verb', 'walk', 'verb', 'run'), ('verb', 'walk', 'verb', 'jump')]
            weights: Optional weights for each direction (default: equal weights)
            aggregate: How to aggregate each sample
        
        Returns:
            Compound direction vector (D,) or (T' * D,) if aggregate='flatten'
        
        Example:
            # Running-and-jumping direction: walk→run + walk→jump
            compound_dir = transformer.compute_compound_direction([
                ('verb', 'walk', 'verb', 'run'),
                ('verb', 'walk', 'verb', 'jump')
            ])
        """
        if not directions:
            raise ValueError("At least one direction must be provided")
        
        # Compute individual direction vectors
        direction_vectors = []
        for source_type, source_id, target_type, target_id in directions:
            direction = self.compute_direction_vector(
                source_type, source_id, target_type, target_id, aggregate
            )
            direction_vectors.append(direction)
        
        # Combine with weights
        if weights is None:
            weights = [1.0] * len(direction_vectors)
        
        if len(weights) != len(direction_vectors):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of directions ({len(direction_vectors)})")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum() * len(weights)  # Normalize but keep relative magnitudes
        
        # Weighted sum
        compound = np.zeros_like(direction_vectors[0])
        for w, dv in zip(weights, direction_vectors):
            compound += w * dv
        
        return compound
    
    def apply_single_direction_intervention(self,
                                           source_embedding: np.ndarray,
                                           direction_vector: np.ndarray,
                                           alpha: float) -> np.ndarray:
        """
        Apply single direction intervention to a source embedding sequence.
        
        z_e^shifted(x_0)_t = z_e(x_0)_t + α · v_{g1→g2}  ∀ t ∈ {1, ..., T'}
        
        Args:
            source_embedding: Source motion embedding z_e(x_0) of shape (T', D)
            direction_vector: Direction vector v_{g1→g2} of shape (D,)
            alpha: Intervention strength
        
        Returns:
            Shifted embedding z_e^shifted(x_0) of shape (T', D)
        """
        if len(source_embedding.shape) == 1:
            # If source is aggregated, broadcast direction
            return source_embedding + alpha * direction_vector
        elif len(source_embedding.shape) == 2:
            # Source is (T', D), direction should be (D,)
            if len(direction_vector.shape) == 1:
                # Broadcast direction to all temporal positions
                return source_embedding + alpha * direction_vector[None, :]
            elif len(direction_vector.shape) == 2:
                # Direction is also (T', D) - apply element-wise
                if source_embedding.shape != direction_vector.shape:
                    raise ValueError(
                        f"Shape mismatch: source {source_embedding.shape} vs "
                        f"direction {direction_vector.shape}"
                    )
                return source_embedding + alpha * direction_vector
            else:
                raise ValueError(f"Direction vector must be 1D or 2D, got {len(direction_vector.shape)}D")
        else:
            raise ValueError(f"Source embedding must be 1D or 2D, got {len(source_embedding.shape)}D")
    
    def apply_intervention(self,
                          source_sample_id: str,
                          direction_vector: np.ndarray,
                          alpha: float,
                          use_full_sequence: bool = True) -> np.ndarray:
        """
        Apply intervention to a source motion sample.
        
        Args:
            source_sample_id: ID of source motion sample
            direction_vector: Direction vector v_{g1→g2} or v_compound
            alpha: Intervention strength (test with α ∈ {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2})
            use_full_sequence: If True, use full sequence (T', D); if False, use aggregated (D,)
        
        Returns:
            Shifted embedding z_e^shifted(x_0) of shape (T', D) or (D,)
        """
        if use_full_sequence:
            source_emb = self.get_embedding(source_sample_id, aggregate=None)  # (T', D)
        else:
            source_emb = self.get_embedding(source_sample_id, aggregate='mean')  # (D,)
        
        return self.apply_single_direction_intervention(source_emb, direction_vector, alpha)
    
    def apply_intervention_batch(self,
                                source_sample_id: str,
                                direction_vector: np.ndarray,
                                alphas: List[float],
                                use_full_sequence: bool = True) -> Dict[float, np.ndarray]:
        """
        Apply intervention with multiple alpha values.
        
        Args:
            source_sample_id: ID of source motion sample
            direction_vector: Direction vector v_{g1→g2} or v_compound
            alphas: List of intervention strengths to test
            use_full_sequence: If True, use full sequence; if False, use aggregated
        
        Returns:
            Dictionary mapping alpha values to shifted embeddings
        """
        results = {}
        for alpha in alphas:
            results[alpha] = self.apply_intervention(
                source_sample_id, direction_vector, alpha, use_full_sequence
            )
        return results
    
    def save_shifted_embeddings(self,
                               shifted_embeddings: Dict[float, np.ndarray],
                               source_sample_id: str,
                               direction_description: str,
                               save_path: str):
        """
        Save shifted embeddings to HDF5 file for later quantization and decoding.
        
        The saved file can be used with the VQ-VAE decoder:
        - Load shifted embeddings
        - Apply quantization: Q(z_e^shifted)
        - Decode: D(Q(z_e^shifted))
        
        Args:
            shifted_embeddings: Dictionary mapping alpha values to shifted embeddings
            source_sample_id: ID of source sample
            direction_description: Description of the direction (e.g., "walk_to_run")
            save_path: Path to save HDF5 file
        """
        print(f"Saving shifted embeddings to {save_path}...")
        
        with h5py.File(save_path, 'w') as f:
            # Root attributes
            f.attrs['source_sample_id'] = source_sample_id
            f.attrs['direction_description'] = direction_description
            f.attrs['n_interventions'] = len(shifted_embeddings)
            f.attrs['embedding_dim'] = list(shifted_embeddings.values())[0].shape[-1]
            
            # Create group for each alpha value
            for alpha, shifted_emb in shifted_embeddings.items():
                group_name = f"alpha_{alpha:.1f}".replace('-', 'neg').replace('.', 'p')
                group = f.create_group(group_name)
                group.attrs['alpha'] = float(alpha)
                group.create_dataset(
                    'shifted_embeddings',
                    data=shifted_emb,
                    compression='gzip',
                    compression_opts=4
                )
        
        print(f"✓ Saved {len(shifted_embeddings)} shifted embeddings")
    
    def decode_shifted_embeddings(self,
                                 shifted_embeddings: Dict[float, np.ndarray],
                                 decoder: Optional[object] = None,
                                 checkpoint_path: str = "pretrained/VQVAE/net_last.pth",
                                 dataname: str = 't2m',
                                 device: str = 'cuda') -> Dict[float, np.ndarray]:
        """
        Decode shifted embeddings to motion sequences.
        
        This implements: x̂_α = D(Q(z_e^shifted(x_0)))
        
        Args:
            shifted_embeddings: Dictionary mapping alpha values to shifted embeddings
            decoder: Optional LatentDecoder instance (will be created if None)
            checkpoint_path: Path to VQ-VAE checkpoint (used if decoder is None)
            dataname: Dataset name ('t2m' or 'kit')
            device: Device to run on
        
        Returns:
            Dictionary mapping alpha values to decoded motion sequences
        """
        # Import decoder (lazy import to avoid circular dependencies)
        try:
            from decode_latents import LatentDecoder
        except ImportError:
            raise ImportError(
                "decode_latents module not found. "
                "Make sure decode_latents.py is in the same directory."
            )
        
        # Create decoder if not provided
        if decoder is None:
            decoder = LatentDecoder(checkpoint_path, dataname, device)
        
        print(f"Decoding {len(shifted_embeddings)} shifted embeddings...")
        
        decoded_motions = {}
        for alpha, shifted_emb in shifted_embeddings.items():
            print(f"  Decoding α={alpha:.1f}...")
            motion = decoder.quantize_and_decode(shifted_emb)
            decoded_motions[alpha] = motion
            print(f"    Motion shape: {motion.shape}")
        
        print(f"✓ Decoded {len(decoded_motions)} motions")
        return decoded_motions
    
    def get_info(self) -> Dict:
        """Get information about the loaded embeddings."""
        return {
            'n_samples': len(self.embeddings),
            'embedding_dim': self.embedding_dim,
            'n_clusters': self.n_clusters,
            'available_clusters': list(self.cluster_to_samples.keys()),
            'available_verbs': list(self.verb_to_samples.keys()),
            'sample_ids': list(self.embeddings.keys())[:10]  # First 10
        }


# Convenience functions for common operations
def load_transformer(hdf5_path: str = "clustering/outputs/embeddings_clustered_denoised.h5",
                     use_quantized: bool = False) -> LatentSpaceTransformer:
    """Load and return a LatentSpaceTransformer instance."""
    return LatentSpaceTransformer(hdf5_path, use_quantized)


def get_standard_alphas() -> List[float]:
    """
    Get standard alpha values for testing intervention strength.
    
    Returns:
        List of alpha values: [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    """
    return [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]


def example_usage():
    """Example usage of the linear transformation utilities."""
    print("=" * 80)
    print("Linear Transformations in VQ-VAE Latent Space")
    print("=" * 80)
    
    # Load transformer
    transformer = load_transformer()
    
    # Print info
    info = transformer.get_info()
    print(f"\nLoaded {info['n_samples']} samples")
    print(f"Embedding dimension: {info['embedding_dim']}")
    print(f"Available clusters: {info['n_clusters']}")
    print(f"Available verbs: {info['available_verbs'][:10]}...")  # First 10
    
    # Example 1: Single Direction Intervention
    print("\n" + "-" * 80)
    print("Example 1: Single Direction Intervention (walk → run)")
    print("-" * 80)
    
    if 'walk' in info['available_verbs'] and 'run' in info['available_verbs']:
        # Compute direction vector: v_{walk→run}
        direction = transformer.compute_direction_vector(
            'verb', 'walk', 'verb', 'run', aggregate='mean'
        )
        print(f"Direction vector shape: {direction.shape}")
        print(f"Direction vector norm: {np.linalg.norm(direction):.4f}")
        
        # Get a source sample (prefer one labeled as 'walk')
        walk_samples = transformer.verb_to_samples.get('walk', [])
        if walk_samples:
            source_id = walk_samples[0]
            print(f"\nSource sample: {source_id}")
            
            # Test with standard alpha values
            alphas = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            interventions = transformer.apply_intervention_batch(
                source_id, direction, alphas, use_full_sequence=True
            )
            
            print(f"\nApplied interventions with α ∈ {alphas}")
            for alpha, shifted_emb in interventions.items():
                print(f"  α={alpha:4.1f}: shape={shifted_emb.shape}")
    
    # Example 2: Compound Direction Intervention
    print("\n" + "-" * 80)
    print("Example 2: Compound Direction Intervention (running-and-jumping)")
    print("-" * 80)
    
    if all(v in info['available_verbs'] for v in ['walk', 'run', 'jump']):
        # Compute compound direction: v_{walk→run} + v_{walk→jump}
        compound_direction = transformer.compute_compound_direction([
            ('verb', 'walk', 'verb', 'run'),
            ('verb', 'walk', 'verb', 'jump')
        ], weights=[1.0, 1.0], aggregate='mean')
        
        print(f"Compound direction shape: {compound_direction.shape}")
        print(f"Compound direction norm: {np.linalg.norm(compound_direction):.4f}")
        
        # Apply to a source sample
        walk_samples = transformer.verb_to_samples.get('walk', [])
        if walk_samples:
            source_id = walk_samples[0]
            print(f"\nSource sample: {source_id}")
            
            # Apply with alpha=1.0
            shifted = transformer.apply_intervention(
                source_id, compound_direction, alpha=1.0, use_full_sequence=True
            )
            print(f"Shifted embedding shape: {shifted.shape}")
    
    # Example 3: Get verb centroids
    print("\n" + "-" * 80)
    print("Example 3: Get verb centroids")
    print("-" * 80)
    
    if 'walk' in info['available_verbs']:
        walk_centroid = transformer.get_verb_centroid('walk', aggregate='mean')
        print(f"Walk centroid shape: {walk_centroid.shape}")
    
    if 'run' in info['available_verbs']:
        run_centroid = transformer.get_verb_centroid('run', aggregate='mean')
        print(f"Run centroid shape: {run_centroid.shape}")
    
    # Example 4: Cluster centroids
    print("\n" + "-" * 80)
    print("Example 4: Cluster centroids")
    print("-" * 80)
    
    if info['available_clusters']:
        cluster_id = info['available_clusters'][0]
        cluster_centroid = transformer.get_cluster_centroid(cluster_id, aggregate='mean')
        print(f"Cluster {cluster_id} centroid shape: {cluster_centroid.shape}")
        print(f"Number of samples in cluster: {len(transformer.cluster_to_samples[cluster_id])}")
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()
