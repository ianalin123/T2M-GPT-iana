import os
import sys
import argparse
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import seaborn as sns

# new imports
import h5py
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# Ensure project root is on the path so we can import project modules if needed
# Since we're now in clustering/, go up one level to get to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def elbow_method(embeddings, max_clusters=30, aggregate="mean", dim_reduction="pca", n_components=50, save_dir=None, lengths=None):
    """
    Apply the elbow method to determine optimal number of clusters.

    Args:
        embeddings: (N, T', D) array of encoder embeddings or (N, T, 263) raw motion sequences
        max_clusters: maximum number of clusters to test
        aggregate: 'mean', 'max', 'flatten', or None
        dim_reduction: 'pca', 'umap', or None
        n_components: number of dimensions for reduction (None to skip)
        save_dir: directory to save elbow plot (None to skip saving)

    Returns:
        inertias: list of within-cluster sum of squares for each k
        silhouette_scores: list of silhouette scores for each k
        k_range: range of k values tested
        suggested_k: suggested k based on second derivative
        best_silhouette_k: k with best silhouette score
    """
    print(f"\nOriginal embeddings shape: {embeddings.shape}")

    # Handle variable-length sequences if lengths are provided
    if lengths is not None:
        # Mask out padding for proper aggregation
        max_len = embeddings.shape[1]
        mask = np.arange(max_len)[None, :] < lengths[:, None]  # (N, T)
        mask = mask[:, :, None]  # (N, T, 1) for broadcasting
        
        if aggregate == "mean":
            masked_embeddings = embeddings * mask
            embeddings_agg = masked_embeddings.sum(axis=1) / lengths[:, None]  # (N, D)
            print(f"Aggregated with mean pooling (masked): {embeddings_agg.shape}")
        elif aggregate == "max":
            masked_embeddings = embeddings.copy()
            masked_embeddings[~mask.squeeze(-1)] = -np.inf
            embeddings_agg = masked_embeddings.max(axis=1)
            print(f"Aggregated with max pooling (masked): {embeddings_agg.shape}")
        elif aggregate == "flatten":
            embeddings_agg = embeddings.reshape(embeddings.shape[0], -1)
            print(f"Flattened (with padding): {embeddings_agg.shape}")
        else:
            embeddings_agg = embeddings
    else:
        # All sequences have same length
        if aggregate == "mean":
            embeddings_agg = embeddings.mean(axis=1)  # (N, D)
            print(f"Aggregated with mean pooling: {embeddings_agg.shape}")
        elif aggregate == "max":
            embeddings_agg = embeddings.max(axis=1)
            print(f"Aggregated with max pooling: {embeddings_agg.shape}")
        elif aggregate == "flatten":
            embeddings_agg = embeddings.reshape(embeddings.shape[0], -1)
            print(f"Flattened: {embeddings_agg.shape}")
        else:
            embeddings_agg = embeddings

    # Dimensionality reduction
    if n_components and embeddings_agg.shape[1] > n_components:
        if dim_reduction == "pca":
            print(f"\nReducing dimensionality with PCA to {n_components}...")
            reducer = PCA(n_components=n_components)
            embeddings_agg = reducer.fit_transform(embeddings_agg)
            print(f"PCA explained variance: {reducer.explained_variance_ratio_.sum():.2%}")
        elif dim_reduction == "umap":
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP is not installed. Please install it with: pip install umap-learn")
            print(f"\nReducing dimensionality with UMAP to {n_components}...")
            reducer = umap.UMAP(n_components=n_components, random_state=42, verbose=True)
            embeddings_agg = reducer.fit_transform(embeddings_agg)
            print(f"UMAP reduction complete: {embeddings_agg.shape}")

    # Test different numbers of clusters
    k_range = range(2, max_clusters + 1)
    inertias = []
    silhouette_scores = []

    print(f"\nTesting k from 2 to {max_clusters}...")
    for k in k_range:
        print(f"  k={k}...", end=" ", flush=True)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_agg)
        inertias.append(kmeans.inertia_)

        # Calculate silhouette score (only for reasonable k values to save time)
        if k <= 20 or k % 5 == 0:
            from sklearn.metrics import silhouette_score
            score = silhouette_score(embeddings_agg, labels, sample_size=min(10000, len(embeddings_agg)))
            silhouette_scores.append(score)
            print(f"inertia={kmeans.inertia_:.2f}, silhouette={score:.3f}")
        else:
            silhouette_scores.append(None)
            print(f"inertia={kmeans.inertia_:.2f}")

    # Plot elbow curve
    _fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Inertia (within-cluster sum of squares)
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=12)
    axes[0].set_title('Elbow Method: Inertia vs Number of Clusters', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', which='major', labelsize=10)

    # Plot 2: Silhouette score
    valid_k = [k for k, s in zip(k_range, silhouette_scores) if s is not None]
    valid_scores = [s for s in silhouette_scores if s is not None]
    axes[1].plot(valid_k, valid_scores, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, 'elbow_method.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved elbow plot to {plot_path}")

    plt.close()

    # Find potential elbow point using the rate of change
    inertia_diff = np.diff(inertias)
    inertia_diff2 = np.diff(inertia_diff)
    suggested_k = np.argmax(inertia_diff2) + 3  # +3 because we start from k=2 and took 2 diffs

    print(f"\nSuggested k based on second derivative: {suggested_k}")
    best_silhouette_k = None
    if valid_scores:
        best_silhouette_k = valid_k[np.argmax(valid_scores)]
        print(f"Best k based on silhouette score: {best_silhouette_k} (score: {max(valid_scores):.3f})")

    return inertias, silhouette_scores, k_range, suggested_k, best_silhouette_k


def cluster_embeddings(embeddings, n_clusters=20, aggregate="mean", dim_reduction="pca", n_components=50, algorithm="kmeans", min_cluster_size=15, lengths=None):
    """
    Cluster the continuous encoder embeddings or raw motion sequences using various algorithms.

    Args:
        embeddings: (N, T', D) array of encoder embeddings or (N, T, 263) raw motion sequences
        n_clusters: number of clusters (ignored for HDBSCAN)
        aggregate: 'mean', 'max', 'flatten', or None
        dim_reduction: 'pca', 'umap', or None
        n_components: number of dimensions for reduction (None to skip)
        algorithm: 'kmeans', 'gmm', or 'hdbscan'
        min_cluster_size: minimum cluster size for HDBSCAN (default: 15)

    Returns:
        labels: (N,) cluster assignments
        model: fitted clustering model
        embeddings_agg: (N, D) aggregated embeddings used for clustering
        reducer: fitted dimensionality reduction model or None
    """
    print(f"\nOriginal embeddings shape: {embeddings.shape}")

    # Handle variable-length sequences if lengths are provided
    if lengths is not None:
        # Mask out padding for proper aggregation
        # Create a mask: True for valid timesteps, False for padding
        max_len = embeddings.shape[1]
        mask = np.arange(max_len)[None, :] < lengths[:, None]  # (N, T)
        mask = mask[:, :, None]  # (N, T, 1) for broadcasting
        
        # Aggregate over time dimension with masking
        if aggregate == "mean":
            # Sum valid timesteps and divide by actual length
            masked_embeddings = embeddings * mask
            embeddings_agg = masked_embeddings.sum(axis=1) / lengths[:, None]  # (N, D)
            print(f"Aggregated with mean pooling (masked): {embeddings_agg.shape}")
        elif aggregate == "max":
            # Set padding to -inf before max pooling
            masked_embeddings = embeddings.copy()
            masked_embeddings[~mask.squeeze(-1)] = -np.inf
            embeddings_agg = masked_embeddings.max(axis=1)
            print(f"Aggregated with max pooling (masked): {embeddings_agg.shape}")
        elif aggregate == "flatten":
            # Flatten but only use valid timesteps
            # For variable lengths, we need to pad to max length or use a fixed-size representation
            embeddings_agg = embeddings.reshape(embeddings.shape[0], -1)
            print(f"Flattened (with padding): {embeddings_agg.shape}")
        else:
            embeddings_agg = embeddings
    else:
        # All sequences have same length (or we assume they do)
        if aggregate == "mean":
            embeddings_agg = embeddings.mean(axis=1)  # (N, D)
            print(f"Aggregated with mean pooling: {embeddings_agg.shape}")
        elif aggregate == "max":
            embeddings_agg = embeddings.max(axis=1)
            print(f"Aggregated with max pooling: {embeddings_agg.shape}")
        elif aggregate == "flatten":
            embeddings_agg = embeddings.reshape(embeddings.shape[0], -1)
            print(f"Flattened: {embeddings_agg.shape}")
        else:
            embeddings_agg = embeddings

    # Dimensionality reduction
    reducer = None
    if n_components and embeddings_agg.shape[1] > n_components:
        if dim_reduction == "pca":
            print(f"\nReducing dimensionality with PCA to {n_components}...")
            reducer = PCA(n_components=n_components)
            embeddings_agg = reducer.fit_transform(embeddings_agg)
            print(f"PCA explained variance: {reducer.explained_variance_ratio_.sum():.2%}")
        elif dim_reduction == "umap":
            print(f"\nReducing dimensionality with UMAP to {n_components}...")
            reducer = umap.UMAP(n_components=n_components, random_state=42, verbose=True)
            embeddings_agg = reducer.fit_transform(embeddings_agg)
            print(f"UMAP reduction complete: {embeddings_agg.shape}")

    # Clustering
    if algorithm == "kmeans":
        print(f"\nClustering with K-Means into {n_clusters} clusters...")
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1)
        labels = model.fit_predict(embeddings_agg)
    
    elif algorithm == "gmm":
        print(f"\nClustering with Gaussian Mixture Model into {n_clusters} components...")
        model = GaussianMixture(n_components=n_clusters, random_state=42, verbose=1, verbose_interval=10)
        labels = model.fit_predict(embeddings_agg)
        print(f"GMM converged: {model.converged_}")
        print(f"GMM BIC: {model.bic(embeddings_agg):.2f}, AIC: {model.aic(embeddings_agg):.2f}")
    
    elif algorithm == "hdbscan":
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN is not installed. Please install it with: pip install hdbscan")
        print(f"\nClustering with HDBSCAN (min_cluster_size={min_cluster_size})...")
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=None,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        labels = model.fit_predict(embeddings_agg)
        
        # HDBSCAN uses -1 for noise points
        n_noise = np.sum(labels == -1)
        n_clusters_found = len(np.unique(labels[labels != -1]))
        print(f"HDBSCAN found {n_clusters_found} clusters")
        if n_noise > 0:
            print(f"  Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")

    # Print cluster distribution
    print("\nCluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        if label == -1:
            print(f"  Noise   : {count:5d} samples ({count/len(labels)*100:5.1f}%)")
        else:
            print(f"  Cluster {label:2d}: {count:5d} samples ({count/len(labels)*100:5.1f}%)")

    return labels, model, embeddings_agg, reducer


def visualize_clusters(
    embeddings,
    labels,
    _texts,
    save_dir,
    max_samples=100000,
    cluster_verb_labels=None,
    dim_reduction_method=None
):

    # --------------------------------------------------------
    # Global plotting style for publication-quality output
    # --------------------------------------------------------
    style_options = ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "seaborn"]
    style_set = False
    for style_name in style_options:
        try:
            plt.style.use(style_name)
            style_set = True
            break
        except OSError:
            continue
    
    if not style_set:
        # Fallback to default style if seaborn styles aren't available
        plt.style.use("default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "axes.linewidth": 1.0,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 300
    })

    # --------------------------------------------------------
    # Subsample if needed
    # --------------------------------------------------------
    if len(embeddings) > max_samples:
        print(f"\nSubsampling {max_samples} points for visualization...")
        idx = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings_vis = embeddings[idx]
        labels_vis = labels[idx]
    else:
        embeddings_vis = embeddings
        labels_vis = labels

    unique_labels = np.unique(labels_vis)
    n_clusters = len(unique_labels)

    # --------------------------------------------------------
    # Strong, perceptually uniform color palette (HUSL)
    # Works great for <= 30 clusters
    # --------------------------------------------------------
    non_noise_labels = [cid for cid in unique_labels if cid != -1]
    colors = sns.color_palette("husl", len(non_noise_labels))
    cluster_id_to_color_idx = {
        cid: i for i, cid in enumerate(non_noise_labels)
    }

    # --------------------------------------------------------
    # Reusable plotting function
    # --------------------------------------------------------
    def plot_clusters(embeddings_2d, xlabel, ylabel, title, save_path, labels_to_plot=None):
        if labels_to_plot is None:
            labels_to_plot = labels_vis

        fig, ax = plt.subplots(figsize=(10, 8))

        unique_plot_labels = np.unique(labels_to_plot)

        for cid in unique_plot_labels:
            mask = labels_to_plot == cid

            if cid == -1:
                cluster_label = "Noise"
                point_color = (0.75, 0.75, 0.75)
            else:
                # Verb label if provided
                if cluster_verb_labels and cid in cluster_verb_labels:
                    v = cluster_verb_labels[cid]
                    cluster_label = v if v else f"Cluster {cid}"
                else:
                    cluster_label = f"Cluster {cid}"

                color_idx = cluster_id_to_color_idx.get(cid, 0)
                point_color = colors[color_idx]

            # Clean, high-quality scatter
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                s=6,
                color=point_color,
                alpha=0.8,
                label=cluster_label
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=12)    # Remove ticks & grid lines
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        # Legend outside figure
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            title="Clusters",
            title_fontsize=13
        )

        ax.margins(0.05)
        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", metadata={'Creator': None})
        plt.savefig(save_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight", metadata={'Creator': None})
        print(f"Saved: {save_path}")
        print(f"Saved: {save_path.replace('.png', '.pdf')}")
        plt.close()

    # --------------------------------------------------------
    # PCA 2D
    # --------------------------------------------------------
    print("\nComputing PCA for visualization...")
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(embeddings_vis)

    plot_clusters(
        emb_pca,
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        "PCA (2D Projection)",
        os.path.join(save_dir, "clusters_pca.png"),
    )

    # --------------------------------------------------------
    # UMAP (if embeddings already reduced)
    # --------------------------------------------------------
    if dim_reduction_method == "umap" and embeddings_vis.shape[1] >= 2:
        print("\nUsing 2D UMAP embeddings for visualization...")
        emb_umap = embeddings_vis[:, :2]

        plot_clusters(
            emb_umap,
            "UMAP-1",
            "UMAP-2",
            "UMAP (2D Projection)",
            os.path.join(save_dir, "clusters_umap.png"),
        )

    # --------------------------------------------------------
    # t-SNE (limited sample)
    # --------------------------------------------------------
    print("\nComputing t-SNE for visualization...")
    from sklearn.manifold import TSNE

    max_tsne_samples = min(10000, len(embeddings_vis))
    if len(embeddings_vis) > max_tsne_samples:
        print(f"  Subsampling to {max_tsne_samples} points for t-SNE...")
        idx_tsne = np.random.choice(len(embeddings_vis), max_tsne_samples, replace=False)
        emb_input_tsne = embeddings_vis[idx_tsne]
        labels_tsne = labels_vis[idx_tsne]
    else:
        emb_input_tsne = embeddings_vis
        labels_tsne = labels_vis

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        n_iter=1000,
        verbose=1
    )
    emb_tsne = tsne.fit_transform(emb_input_tsne)

    plot_clusters(
        emb_tsne,
        "t-SNE-1",
        "t-SNE-2",
        "t-SNE (2D Projection)",
        os.path.join(save_dir, "clusters_tsne.png"),
        labels_to_plot=labels_tsne,
    )


def load_raw_motion_from_numpy(data_dir):
    """
    Load raw motion sequences from numpy files.
    
    Args:
        data_dir: Directory containing motions.npy, texts.txt, names.txt, and optionally lengths.npy
    
    Returns:
        dict with:
            - motions: (N, T, 263) numpy array of raw motion sequences
            - texts: list of text descriptions
            - names: list of sample IDs
            - lengths: (N,) numpy array of actual sequence lengths (if available)
    """
    print(f"Loading raw motion data from {data_dir}")
    
    motions_path = os.path.join(data_dir, "motions.npy")
    if not os.path.exists(motions_path):
        raise FileNotFoundError(f"motions.npy not found in {data_dir}")
    
    motions = np.load(motions_path)
    print(f"  Loaded motions: {motions.shape}")
    
    # Load metadata
    texts_path = os.path.join(data_dir, "texts.txt")
    names_path = os.path.join(data_dir, "names.txt")
    lengths_path = os.path.join(data_dir, "lengths.npy")
    
    if not os.path.exists(texts_path):
        raise FileNotFoundError(f"texts.txt not found in {data_dir}")
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"names.txt not found in {data_dir}")
    
    with open(texts_path, "r") as f:
        texts = f.read().strip().split("\n")
    with open(names_path, "r") as f:
        names = f.read().strip().split("\n")
    
    lengths = None
    if os.path.exists(lengths_path):
        lengths = np.load(lengths_path)
        print(f"  Loaded lengths: {lengths.shape}")
        print(f"  Sequence length range: {lengths.min()} - {lengths.max()}")
    else:
        print(f"  Warning: lengths.npy not found. Assuming all sequences have same length.")
    
    # Validate dimensions
    if len(texts) != len(motions):
        raise ValueError(f"Mismatch: {len(texts)} texts but {len(motions)} motion sequences")
    if len(names) != len(motions):
        raise ValueError(f"Mismatch: {len(names)} names but {len(motions)} motion sequences")
    if lengths is not None and len(lengths) != len(motions):
        raise ValueError(f"Mismatch: {len(lengths)} lengths but {len(motions)} motion sequences")
    
    return {
        "motions": motions,
        "texts": texts,
        "names": names,
        "lengths": lengths,
    }


def load_data_from_hdf5(hdf5_path, use_quantized=False):
    """
    Load embeddings and metadata from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 embeddings file
        use_quantized: If True, use quantized_embeddings instead of encoder_embeddings

    Returns:
        dict with:
            - encoder_embeddings: (N, T', D) numpy array (or quantized if use_quantized=True)
            - code_indices: (N, T') numpy array
            - texts: list of text descriptions
            - names: list of sample IDs
            - compound_verbs: list of compound verb labels (e.g., "walk-run-sit")
    """
    print(f"Loading data from HDF5: {hdf5_path}")
    
    embedding_key = "quantized_embeddings" if use_quantized else "encoder_embeddings"
    print(f"Using '{embedding_key}' for clustering")

    embeddings = []
    code_indices = []
    texts = []
    names = []
    compound_verbs = []

    with h5py.File(hdf5_path, "r") as f:
        for sample_id in f.keys():
            sample_group = f[sample_id]

            embeddings.append(sample_group[embedding_key][:])
            code_indices.append(sample_group["code_indices"][:])
            texts.append(sample_group.attrs["text"])
            names.append(sample_id)
            compound_verbs.append(sample_group.attrs["compound_verb"])

    return {
        "encoder_embeddings": np.array(embeddings),  # Keep name for compatibility
        "code_indices": np.array(code_indices),
        "texts": texts,
        "names": names,
        "compound_verbs": compound_verbs,
    }


def label_clusters_by_majority_verb(labels, compound_verbs):
    """
    Label each cluster by the most frequent atomic verb within it.

    For each cluster:
    1. Collect all compound verb labels (e.g., "walk-run-sit")
    2. Break them into atomic verbs
    3. Find the most frequent atomic verb across all samples in that cluster

    Args:
        labels: (N,) cluster assignments (can include -1 for HDBSCAN noise)
        compound_verbs: list of compound verb strings (e.g., ["walk-run", "sit", ...])

    Returns:
        cluster_verb_labels: dict mapping cluster_id to most frequent atomic verb
        cluster_atomic_verb_counts: dict mapping cluster_id to Counter of atomic verbs
        cluster_compound_verb_counts: dict mapping cluster_id to Counter of compound verbs
    """
    if compound_verbs is None:
        return None, None, None

    cluster_atomic_verb_counts = defaultdict(Counter)
    cluster_compound_verb_counts = defaultdict(Counter)

    # Count verbs for each cluster
    for i, cluster_id in enumerate(labels):
        # Skip noise points for HDBSCAN (labeled as -1)
        if cluster_id == -1:
            continue
            
        compound_verb = compound_verbs[i] if i < len(compound_verbs) else ""

        # Count the compound verb itself
        cluster_compound_verb_counts[cluster_id][compound_verb] += 1

        # Break into atomic verbs and count each one
        if compound_verb:
            atomic_verbs = compound_verb.split("-")
            for verb in atomic_verbs:
                cluster_atomic_verb_counts[cluster_id][verb] += 1

    # Find most frequent atomic verb for each cluster (excluding noise)
    cluster_verb_labels = {}
    unique_clusters = [c for c in np.unique(labels) if c != -1]
    for cluster_id in unique_clusters:
        atomic_counts = cluster_atomic_verb_counts[cluster_id]
        if not atomic_counts:
            cluster_verb_labels[cluster_id] = "unknown"
            continue

        # 1) get top frequency
        most_common = atomic_counts.most_common()
        max_count = most_common[0][1]
        tied_verbs = [v for v, c in most_common if c == max_count]

        if len(tied_verbs) == 1:
            # no tie
            cluster_verb_labels[cluster_id] = tied_verbs[0]
        else:
            # 2) tie-break: prefer verb that appears most often as a pure compound label
            comp_counts = cluster_compound_verb_counts[cluster_id]
            best_verb = None
            best_pure_count = -1
            for v in tied_verbs:
                pure_count = comp_counts.get(v, 0)  # counts of "v" as the whole compound_verb
                if pure_count > best_pure_count:
                    best_pure_count = pure_count
                    best_verb = v

            # 3) if still tied (all zero), fall back to deterministic choice
            if best_verb is None:
                best_verb = sorted(tied_verbs)[0]

            cluster_verb_labels[cluster_id] = best_verb

    return cluster_verb_labels, cluster_atomic_verb_counts, cluster_compound_verb_counts


def save_clustered_hdf5(data, labels, cluster_verb_labels, save_path):
    """
    Save clustered data to HDF5 file with cluster assignments and atomic verb labels.

    Structure:
        /[file_id]/
            compound_verb (attribute) - original compound verb from verbs.txt
            cluster_id (attribute) - k-means cluster assignment
            cluster_label (attribute) - atomic verb label for this cluster
            text (attribute)
            length (attribute)
            encoder_embeddings (dataset)
            code_indices (dataset)

    Args:
        data: dict with encoder_embeddings, code_indices, texts, names, compound_verbs
        labels: (N,) k-means cluster assignments
        cluster_verb_labels: dict mapping cluster_id to atomic verb label
        save_path: Path to save clustered HDF5 file
    """
    print(f"\nSaving clustered data to HDF5: {save_path}")

    with h5py.File(save_path, "w") as f:
        # Add metadata attributes at root level
        f.attrs["n_samples"] = len(data["names"])
        f.attrs["n_clusters"] = len(cluster_verb_labels)
        f.attrs["embedding_dim"] = data["encoder_embeddings"].shape[-1]

        # Store cluster verb labels mapping (one attribute per cluster)
        for cluster_id, verb_label in cluster_verb_labels.items():
            f.attrs[f"cluster_{cluster_id}_label"] = verb_label

        # Create a group for each sample
        for idx, name in enumerate(tqdm(data["names"], desc="Writing clustered HDF5")):
            sample_group = f.create_group(name)

            # Add original compound verb
            sample_group.attrs["compound_verb"] = data["compound_verbs"][idx]

            # Add cluster assignment and label
            sample_group.attrs["cluster_id"] = int(labels[idx])
            # Handle noise points for HDBSCAN (-1 cluster)
            if labels[idx] == -1:
                sample_group.attrs["cluster_label"] = "noise"
            elif labels[idx] in cluster_verb_labels:
                sample_group.attrs["cluster_label"] = cluster_verb_labels[labels[idx]]
            else:
                sample_group.attrs["cluster_label"] = "unknown"

            # Add text and length
            sample_group.attrs["text"] = data["texts"][idx]
            sample_group.attrs["length"] = int(data["encoder_embeddings"][idx].shape[0])

            # Add embeddings and codes
            sample_group.create_dataset(
                "encoder_embeddings",
                data=data["encoder_embeddings"][idx],
                compression="gzip",
                compression_opts=4,
            )
            sample_group.create_dataset(
                "code_indices",
                data=data["code_indices"][idx],
                compression="gzip",
                compression_opts=4,
            )

    print(f"✓ Saved {len(data['names'])} samples with cluster assignments to {save_path}")


def analyze_clusters(
    labels, texts, code_indices, save_dir, top_k=10, compound_verbs=None,
    cluster_verb_labels=None, cluster_atomic_verb_counts=None, cluster_compound_verb_counts=None,
    algorithm="kmeans", names=None
):
    """Analyze what each cluster represents."""
    
    # Map algorithm to display name
    algo_display = {
        "kmeans": "K-Means",
        "gmm": "Gaussian Mixture Model",
        "hdbscan": "HDBSCAN"
    }
    algo_name = algo_display.get(algorithm, "K-Means")

    print("\n" + "=" * 80)
    print(f"CLUSTER ANALYSIS ({algo_name.upper()})")
    print("=" * 80)

    # Handle noise points for HDBSCAN
    if algorithm == "hdbscan":
        n_clusters = len(np.unique(labels[labels != -1]))
    else:
        n_clusters = len(np.unique(labels))

    analysis_path = os.path.join(save_dir, "cluster_analysis.txt")
    analysis_file = open(analysis_path, "w")

    # Also save cluster labels
    if cluster_verb_labels is not None:
        cluster_labels_path = os.path.join(save_dir, "cluster_verb_labels.txt")
        with open(cluster_labels_path, "w") as f:
            for cluster_id in sorted(cluster_verb_labels.keys()):
                f.write(f"{cluster_id}\t{cluster_verb_labels[cluster_id]}\n")
        print(f"Saved cluster verb labels to {cluster_labels_path}")

    # Get unique cluster IDs (excluding noise for HDBSCAN)
    if algorithm == "hdbscan":
        cluster_ids = sorted([c for c in np.unique(labels) if c != -1])
    else:
        cluster_ids = range(n_clusters)
    
    # Handle noise cluster separately for HDBSCAN
    if algorithm == "hdbscan" and -1 in labels:
        noise_mask = labels == -1
        noise_texts = [texts[i] for i, m in enumerate(noise_mask) if m]
        noise_indices = [i for i, m in enumerate(noise_mask) if m]
        
        # Save noise samples to file
        noise_samples_path = os.path.join(save_dir, "noise_samples.txt")
        with open(noise_samples_path, "w") as f:
            f.write(f"Noise Samples - {len(noise_texts)} samples\n")
            f.write(f"{'='*80}\n")
            f.write("These samples were not assigned to any cluster by HDBSCAN.\n")
            f.write(f"{'='*80}\n\n")
            
            for idx, (sample_idx, text) in enumerate(zip(noise_indices, noise_texts), 1):
                sample_name = names[sample_idx] if names and sample_idx < len(names) else f"sample_{sample_idx}"
                compound_verb = compound_verbs[sample_idx] if compound_verbs and sample_idx < len(compound_verbs) else "N/A"
                f.write(f"{idx}. Sample: {sample_name}\n")
                f.write(f"   Text: {text}\n")
                f.write(f"   Compound Verb: {compound_verb}\n")
                f.write(f"\n")
        
        print(f"Saved {len(noise_texts)} noise samples to {noise_samples_path}")
        
        analysis = f"\n{'='*80}\n"
        analysis += (
            f"{algo_name} Noise Points - {len(noise_texts)} samples "
            f"({len(noise_texts)/len(labels)*100:.1f}%)\n"
        )
        analysis += f"{'='*80}\n"
        analysis += "\nThese samples were not assigned to any cluster.\n"
        analysis_file.write(analysis)
        print(analysis)

    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        cluster_texts = [texts[i] for i, m in enumerate(mask) if m]

        # Get cluster label
        cluster_label = ""
        if cluster_verb_labels is not None and cluster_id in cluster_verb_labels:
            cluster_label = f" ({cluster_verb_labels[cluster_id]})"

        analysis = f"\n{'='*80}\n"
        analysis += (
            f"{algo_name} Cluster {cluster_id}{cluster_label} - {len(cluster_texts)} samples "
            f"({len(cluster_texts)/len(labels)*100:.1f}%)\n"
        )
        analysis += f"{'='*80}\n"

        # Sample text descriptions
        analysis += f"\nSample descriptions (first {top_k}):\n"
        for i, text in enumerate(cluster_texts[:top_k]):
            analysis += f"  {i+1}. {text}\n"

        # Most common discrete codes used (only if code_indices available)
        if code_indices is not None:
            cluster_codes = code_indices[mask]
            all_codes = cluster_codes.flatten()
            unique_codes, counts = np.unique(all_codes, return_counts=True)
            top_code_indices = np.argsort(-counts)[:10]

            analysis += "\nMost frequently used codebook entries:\n"
            for idx in top_code_indices:
                code = unique_codes[idx]
                count = counts[idx]
                analysis += (
                    f"  Code {code:3d}: used {count:5d} times "
                    f"({count/len(all_codes)*100:.1f}%)\n"
                )

            analysis += (
                f"\nCodebook diversity: {len(unique_codes)}/{code_indices.max()+1} "
                "codes used\n"
            )
        else:
            analysis += "\n(Codebook analysis not available for raw motion data)\n"

        # Atomic verb distribution in this cluster
        if cluster_atomic_verb_counts is not None:
            atomic_counts = cluster_atomic_verb_counts[cluster_id]
            analysis += "\nAtomic verb distribution in this cluster:\n"
            for verb, count in atomic_counts.most_common(15):
                analysis += (
                    f"  {verb:20s}: {count:4d} occurrences\n"
                )

        # Compound verb distribution in this cluster
        if cluster_compound_verb_counts is not None:
            compound_counts = cluster_compound_verb_counts[cluster_id]
            analysis += "\nCompound verb distribution in this cluster (top 15):\n"
            for verb, count in compound_counts.most_common(15):
                verb_display = verb if verb else "(no verb)"
                analysis += (
                    f"  {verb_display:40s}: {count:4d} samples "
                    f"({count/len(cluster_texts)*100:.1f}%)\n"
                )

        print(analysis)
        analysis_file.write(analysis)

    analysis_file.close()
    print(f"\nSaved full analysis to {analysis_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run K-means clustering on encoder embeddings and/or "
            "cluster samples by verb labels."
        )
    )
    # input: either data-dir or hdf5-file
    parser.add_argument(
        "--data-dir",
        default="clustering/outputs",
        type=str,
        help="Directory containing encoder_embeddings.npy, code_indices.npy, texts.txt, names.txt",
    )
    parser.add_argument(
        "--hdf5-file",
        default=None,
        type=str,
        help="Path to HDF5 embeddings file (e.g., embeddings_train.h5). If provided, will use this instead of numpy files.",
    )
    # output
    parser.add_argument(
        "--save-dir",
        default="clustering/outputs",
        type=str,
        help="Directory to save clustering results (defaults to data-dir)",
    )
    parser.add_argument(
        "--dim-reduction",
        default="umap",
        choices=["pca", "umap", "none"],
        help="Dimensionality reduction method: 'pca', 'umap', or 'none' (default: pca)",
    )
    parser.add_argument(
        "--clustering-algorithm",
        default="kmeans",
        choices=["kmeans", "gmm", "hdbscan"],
        help="Clustering algorithm: 'kmeans' (K-Means, default), 'gmm' (Gaussian Mixture Model), or 'hdbscan' (HDBSCAN)",
    )

    ### optional arguments
    parser.add_argument(
        "--aggregate",
        default="mean",
        choices=["mean", "max", "flatten", "none"],
        help="Aggregate method for embeddings: 'mean' (default), 'max', 'flatten', or 'none'",
    )
    
    # use quantized embeddings
    parser.add_argument(
        "--use-quantized",
        action="store_true",
        help="Use quantized embeddings instead of encoder embeddings for clustering (only works with --hdf5-file)",
    )
    # use raw motion sequences
    parser.add_argument(
        "--use-raw-motion",
        action="store_true",
        help="Use raw motion sequences instead of embeddings for clustering (loads from motions.npy in data-dir)",
    )


    # dimensionality reduction (both pca and umap)
    parser.add_argument(
        "--dim-reduction-dims",
        default=50,
        type=int,
        help="Number of dimensions for dimensionality reduction (default: 50)",
    )

    # umap
    parser.add_argument('--umap-n-neighbors', default=15, type=int,
                   help='UMAP n_neighbors parameter (default: 15)',
                )
    parser.add_argument('--umap-min-dist', default=0.1, type=float,
                   help='UMAP min_dist parameter (default: 0.1)',
                )
    parser.add_argument('--umap-metric', default='euclidean', type=str,
                   help='UMAP metric parameter',
                )

    # k-means
    parser.add_argument(
        "--n-clusters",
        default=None,
        type=int,
        help="Number of clusters for K-means. Specify if you want to use a specific k, otherwise auto-selects k using elbow method with derivative.",
    )
    # auto-select k for k-means and gmm
    parser.add_argument(
        "--max-clusters",
        default=30,
        type=int,
        help="Maximum number of clusters to test for elbow method (default: 30)",
    )
    parser.add_argument(
        "--auto-k-method",
        default="derivative",
        choices=["silhouette", "derivative"],
        help="Method to choose optimal k when auto-selecting: 'derivative' (elbow point, default) or 'silhouette' (best silhouette score)",
    )
    # hdbscan
    parser.add_argument(
        "--min-cluster-size",
        default=15,
        type=int,
        help="Minimum cluster size for HDBSCAN (default: 15)",
    )
    parser.add_argument(
        "--min-samples",
        default=None,
        type=int,
        help="Minimum samples for HDBSCAN (default: None --> use min_cluster_size)",
    )
    parser.add_argument(
        "--cluster-selection-epsilon",
        default=0.5,
        type=float,
        help="Cluster selection epsilon for HDBSCAN (default: 0.5)",
    )
    parser.add_argument(
        "--cluster-selection-method",
        default="eom",
        choices=["eom", "leaf", "dbscan"],
        help="Cluster selection method for HDBSCAN: 'eom' (default), 'leaf', or 'dbscan'",
    )

    args = parser.parse_args()

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Load extracted data
    print("=" * 80)
    if args.use_raw_motion:
        print("Loading raw motion sequences and metadata...")
    else:
        print("Loading extracted embeddings and metadata...")
    print("=" * 80)

    # Validate incompatible options
    if args.use_raw_motion and args.use_quantized:
        raise ValueError("--use-raw-motion cannot be used with --use-quantized.")

    # Load from HDF5 or legacy numpy / raw-motion files
    if args.hdf5_file:
        # Validate --use-quantized option
        if args.use_quantized:
            print("\n⚠️  Using quantized embeddings for clustering")
        
        # Load from HDF5
        data = load_data_from_hdf5(args.hdf5_file, use_quantized=args.use_quantized)
        # Choose source for clustering: raw motions vs embeddings
        if args.use_raw_motion and "motions" in data:
            print("\n⚠️  Using raw motion sequences from HDF5 for clustering")
            encoder_embeddings = data["motions"]  # (N, T, 263)
            lengths = data.get("lengths", None)
        else:
            encoder_embeddings = data["encoder_embeddings"]
            lengths = None  # lengths only defined for raw motion path

        code_indices = data["code_indices"]
        texts = data["texts"]
        names = data["names"]
        compound_verbs = data["compound_verbs"]
        
        print(f"  Input for clustering: {encoder_embeddings.shape}")
        if lengths is not None:
            print(f"  Sequence lengths: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}")
        print(f"  Code indices: {code_indices.shape}")
        print(f"  Number of texts: {len(texts)}")
        print(f"  Number of samples: {len(names)}")

    # Fallback: legacy NumPy input (embeddings and optional motions)
    else:
        if args.use_raw_motion:
            print("\n⚠️  Using raw motion sequences from NumPy for clustering")
            data = load_raw_motion_from_numpy(args.data_dir)
            encoder_embeddings = data["motions"]  # (N, T, 263)
            lengths = data["lengths"]
            texts = data["texts"]
            names = data["names"]
            code_indices = None
            compound_verbs = None

            print(f"  Raw motions: {encoder_embeddings.shape}")
            if lengths is not None:
                print(f"  Sequence lengths: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}")
            print(f"  Number of texts: {len(texts)}")
            print(f"  Number of samples: {len(names)}")
        else:
            # Legacy embeddings-only path
            if args.use_quantized:
                print("\n⚠️  WARNING: --use-quantized only works with --hdf5-file. Ignoring flag.")
            
            encoder_embeddings = np.load(os.path.join(args.data_dir, "encoder_embeddings.npy"))
            code_indices = np.load(os.path.join(args.data_dir, "code_indices.npy"))

            with open(os.path.join(args.data_dir, "texts.txt"), "r") as f:
                texts = f.read().strip().split("\n")
            with open(os.path.join(args.data_dir, "names.txt"), "r") as f:
                names = f.read().strip().split("\n")

            compound_verbs = None
            lengths = None
            lengths_path = os.path.join(args.data_dir, "lengths.npy")
            if os.path.exists(lengths_path):
                lengths = np.load(lengths_path)
                print(f"  Loaded sequence lengths: {lengths.shape}")

            print(f"  Encoder embeddings: {encoder_embeddings.shape}")
            print(f"  Code indices: {code_indices.shape}")
            print(f"  Number of texts: {len(texts)}")
            print(f"  Number of samples: {len(names)}")

    # Validate dimensionality reduction settings
    dim_reduction_method = None if args.dim_reduction == "none" else args.dim_reduction

    # Determine if we need to run elbow method for auto k-selection
    # HDBSCAN doesn't need k, so skip elbow method for it
    user_specified_k = args.n_clusters is not None
    run_elbow = not user_specified_k and args.clustering_algorithm != "hdbscan"
    
    optimal_k = args.n_clusters  # Will be None if not specified
    
    if run_elbow:
        print("\n" + "=" * 80)
        print("Running Elbow Method to auto-select k...")
        print(f"Testing k from 2 to 30")
        print("=" * 80)
        _, _, _, suggested_k, best_silhouette_k = elbow_method(
            encoder_embeddings,
            max_clusters=args.max_clusters,
            aggregate="mean", # mean pooling over time dimension
            dim_reduction=dim_reduction_method,
            n_components=args.dim_reduction_dims,
            save_dir=args.save_dir,
            lengths=lengths,
        )

        # Auto-select k based on method
        if args.auto_k_method == "silhouette" and best_silhouette_k is not None:
            optimal_k = best_silhouette_k
            print(f"\n>>> Auto-selected k={optimal_k} based on silhouette score")
        elif args.auto_k_method == "derivative" and suggested_k is not None:
            optimal_k = suggested_k
            print(f"\n>>> Auto-selected k={optimal_k} based on elbow point (derivative method)")
        else:
            optimal_k = 20  # Fallback default
            print(f"\n>>> Could not auto-select k, using fallback k={optimal_k}")
    else:
        print(f"\n>>> Using user-specified k={optimal_k}")

    # Clustering on encoder embeddings or raw motion
    print("\n" + "=" * 80)
    algo_name = {"kmeans": "K-Means", "gmm": "Gaussian Mixture Model", "hdbscan": "HDBSCAN"}[args.clustering_algorithm]
    data_type = "raw motion sequences" if args.use_raw_motion else "embeddings"
    if args.clustering_algorithm == "hdbscan":
        print(f"{algo_name} Clustering {data_type} (min_cluster_size={args.min_cluster_size})...")
    else:
        print(f"{algo_name} Clustering {data_type} with k={optimal_k}...")
    if dim_reduction_method:
        print(f"Using {dim_reduction_method.upper()} for dimensionality reduction to {args.dim_reduction_dims} dimensions")
    print("=" * 80)

    labels, model, embeddings_processed, _ = cluster_embeddings(
        encoder_embeddings,
        n_clusters=optimal_k if optimal_k else 20,  # Fallback for HDBSCAN (ignored anyway)
        aggregate=args.aggregate,
        dim_reduction=dim_reduction_method,
        n_components=args.dim_reduction_dims,
        algorithm=args.clustering_algorithm,
        min_cluster_size=args.min_cluster_size,
        lengths=lengths,
    )

    # Save clustering results
    np.save(os.path.join(args.save_dir, "cluster_labels.npy"), labels)
    np.save(
        os.path.join(args.save_dir, "embeddings_processed.npy"), embeddings_processed
    )

    # Label clusters by finding most frequent atomic verb in each cluster
    print("\n" + "=" * 80)
    print("Labeling clusters by most frequent atomic verb...")
    print("=" * 80)
    cluster_verb_labels, cluster_atomic_verb_counts, cluster_compound_verb_counts = label_clusters_by_majority_verb(
        labels, compound_verbs
    )

    # Print cluster labels
    if cluster_verb_labels:
        print("\nCluster labels (atomic verb):")
        for cluster_id in sorted(cluster_verb_labels.keys()):
            print(f"  Cluster {cluster_id:2d}: {cluster_verb_labels[cluster_id]}")

    # Visualize clusters
    print("\n" + "=" * 80)
    print("Visualizing clusters...")
    print("=" * 80)
    visualize_clusters(
        embeddings_processed, 
        labels, 
        texts, 
        args.save_dir, 
        cluster_verb_labels=cluster_verb_labels,
        dim_reduction_method=dim_reduction_method
    )

    # Analyze clusters
    print("\n" + "=" * 80)
    print(f"Analyzing {args.clustering_algorithm.upper()} clusters...")
    print("=" * 80)
    analyze_clusters(
        labels,
        texts,
        code_indices,
        args.save_dir,
        compound_verbs=compound_verbs,
        cluster_verb_labels=cluster_verb_labels,
        cluster_atomic_verb_counts=cluster_atomic_verb_counts,
        cluster_compound_verb_counts=cluster_compound_verb_counts,
        algorithm=args.clustering_algorithm,
        names=names,
    )
    
    # Note: code_indices may be None for raw motion, but analyze_clusters handles it

    # Save clustered HDF5 with cluster assignments and labels
    if args.hdf5_file:
        print("\n" + "=" * 80)
        print("Saving clustered HDF5 file...")
        print("=" * 80)
        clustered_hdf5_path = os.path.join(args.save_dir, "embeddings_clustered.h5")
        save_clustered_hdf5(
            data,
            labels,
            cluster_verb_labels,
            clustered_hdf5_path,
        )
        print(f"Saved clustered embeddings to {clustered_hdf5_path}")

    print("\n" + "=" * 80)
    print(f"✓ Done! All results saved to {args.save_dir}/")
    print("=" * 80)
    print("\nGenerated files:")
    if run_elbow:
        print("  - elbow_method.png: elbow method plot with inertia and silhouette scores")
    algo_display = {"kmeans": "K-Means", "gmm": "GMM", "hdbscan": "HDBSCAN"}
    algo_name = algo_display.get(args.clustering_algorithm, "K-Means")
    print(f"  - cluster_labels.npy: {algo_name} cluster assignments (N,)")
    if cluster_verb_labels:
        print("  - cluster_verb_labels.txt: atomic verb label for each cluster")
    data_type_desc = "processed motion/embeddings" if args.use_raw_motion else "processed embeddings"
    print(f"  - embeddings_processed.npy: {data_type_desc} used for clustering")
    print("  - clusters_pca.png: PCA visualization with atomic verb labels")
    if dim_reduction_method == "umap":
        print("  - clusters_umap.png: UMAP visualization with atomic verb labels")
    print("  - clusters_tsne.png: t-SNE visualization with atomic verb labels")
    print("  - pca_variance.png: PCA variance analysis")
    print(f"  - cluster_analysis.txt: detailed cluster analysis ({algo_name.upper()}, atomic + compound verb distributions)")
    if args.hdf5_file:
        print("  - embeddings_clustered.h5: HDF5 file with cluster assignments and atomic verb labels")


if __name__ == "__main__":
    main()

