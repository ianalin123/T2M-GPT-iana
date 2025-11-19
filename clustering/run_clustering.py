import os
import sys
import argparse
from collections import defaultdict, Counter

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

# Ensure project root is on the path so we can import project modules if needed
# Since we're now in clustering/, go up one level to get to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def elbow_method(embeddings, max_clusters=30, aggregate="mean", pca_dim=50, save_dir=None):
    """
    Apply the elbow method to determine optimal number of clusters.

    Args:
        embeddings: (N, T', D) array of encoder embeddings
        max_clusters: maximum number of clusters to test
        aggregate: 'mean', 'max', 'flatten', or None
        pca_dim: dimensionality reduction (None to skip)
        save_dir: directory to save elbow plot (None to skip saving)

    Returns:
        inertias: list of within-cluster sum of squares for each k
        silhouette_scores: list of silhouette scores for each k
        k_range: range of k values tested
        suggested_k: suggested k based on second derivative
        best_silhouette_k: k with best silhouette score
    """
    print(f"\nOriginal embeddings shape: {embeddings.shape}")

    # Aggregate over time dimension
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

    # PCA dimensionality reduction
    if pca_dim and embeddings_agg.shape[1] > pca_dim:
        print(f"\nReducing dimensionality with PCA to {pca_dim}...")
        pca = PCA(n_components=pca_dim)
        embeddings_agg = pca.fit_transform(embeddings_agg)
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

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


def cluster_embeddings(embeddings, n_clusters=20, aggregate="mean", pca_dim=50):
    """
    Cluster the continuous encoder embeddings with K-means.

    Args:
        embeddings: (N, T', D) array of encoder embeddings
        n_clusters: number of clusters
        aggregate: 'mean', 'max', 'flatten', or None
        pca_dim: dimensionality reduction (None to skip)

    Returns:
        labels: (N,) cluster assignments
        kmeans: fitted KMeans model
        embeddings_agg: (N, D) aggregated embeddings used for clustering
        pca: fitted PCA model or None
    """
    print(f"\nOriginal embeddings shape: {embeddings.shape}")

    # Aggregate over time dimension
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

    # PCA dimensionality reduction
    if pca_dim and embeddings_agg.shape[1] > pca_dim:
        print(f"\nReducing dimensionality with PCA to {pca_dim}...")
        pca = PCA(n_components=pca_dim)
        embeddings_agg = pca.fit_transform(embeddings_agg)
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    else:
        pca = None

    # K-means clustering
    print(f"\nClustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1)
    labels = kmeans.fit_predict(embeddings_agg)

    # Print cluster distribution
    print("\nCluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  Cluster {label:2d}: {count:5d} samples ({count/len(labels)*100:5.1f}%)")

    return labels, kmeans, embeddings_agg, pca


def visualize_clusters(embeddings, labels, _texts, save_dir, max_samples=100000, cluster_verb_labels=None):
    """Create PCA and t-SNE visualizations of clustered embeddings."""

    # Subsample if too many
    if len(embeddings) > max_samples:
        print(f"\nSubsampling {max_samples} points for visualization...")
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings_vis = embeddings[indices]
        labels_vis = labels[indices]
    else:
        embeddings_vis = embeddings
        labels_vis = labels

    # Get unique clusters and their verb labels
    unique_labels = np.unique(labels_vis)
    n_clusters = len(unique_labels)

    # Create color map
    if n_clusters <= 20:
        cmap = plt.cm.get_cmap("tab20")
    else:
        cmap = plt.cm.get_cmap("tab20b")

    # PCA - Fast and interpretable!
    print("\nComputing PCA for visualization...")
    pca_viz = PCA(n_components=2)
    embeddings_pca = pca_viz.fit_transform(embeddings_vis)

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot each cluster separately for legend
    for cluster_id in unique_labels:
        mask = labels_vis == cluster_id
        if cluster_verb_labels and cluster_id in cluster_verb_labels:
            verb = cluster_verb_labels[cluster_id]
            cluster_label = verb if verb else "(no verb)"
        else:
            cluster_label = f"Cluster {cluster_id}"

        ax.scatter(
            embeddings_pca[mask, 0],
            embeddings_pca[mask, 1],
            c=[cmap(cluster_id / n_clusters)],
            label=cluster_label,
            alpha=0.6,
            s=10,
        )

    ax.set_xlabel(f"PC1 ({pca_viz.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca_viz.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax.set_title(
        "PCA Visualization of Latent Embeddings\n"
        f"(PC1: {pca_viz.explained_variance_ratio_[0]:.1%}, "
        f"PC2: {pca_viz.explained_variance_ratio_[1]:.1%}, "
        f"Total: {pca_viz.explained_variance_ratio_.sum():.1%})",
        fontsize=14,
        fontweight='bold'
    )

    # Add legend
    if n_clusters <= 30:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=8)
    else:
        # For many clusters, use smaller font and multiple columns
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fontsize=6)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "clusters_pca.png"), dpi=300, bbox_inches="tight")
    print(f"Saved PCA visualization to {os.path.join(save_dir, 'clusters_pca.png')}")
    plt.close()

    # t-SNE - Better for visualizing local structure
    print("\nComputing t-SNE for visualization...")
    from sklearn.manifold import TSNE

    # For t-SNE, use fewer samples if needed (it's slower)
    max_tsne_samples = min(10000, len(embeddings_vis))
    if len(embeddings_vis) > max_tsne_samples:
        print(f"  Subsampling to {max_tsne_samples} points for t-SNE...")
        indices_tsne = np.random.choice(len(embeddings_vis), max_tsne_samples, replace=False)
        embeddings_tsne_input = embeddings_vis[indices_tsne]
        labels_tsne = labels_vis[indices_tsne]
    else:
        embeddings_tsne_input = embeddings_vis
        labels_tsne = labels_vis

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=1)
    embeddings_tsne = tsne.fit_transform(embeddings_tsne_input)

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot each cluster separately for legend
    unique_labels_tsne = np.unique(labels_tsne)
    for cluster_id in unique_labels_tsne:
        mask = labels_tsne == cluster_id
        if cluster_verb_labels and cluster_id in cluster_verb_labels:
            verb = cluster_verb_labels[cluster_id]
            cluster_label = verb if verb else "(no verb)"
        else:
            cluster_label = f"Cluster {cluster_id}"

        ax.scatter(
            embeddings_tsne[mask, 0],
            embeddings_tsne[mask, 1],
            c=[cmap(cluster_id / n_clusters)],
            label=cluster_label,
            alpha=0.6,
            s=10,
        )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title("t-SNE Visualization of Latent Embeddings", fontsize=14, fontweight='bold')

    # Add legend
    if n_clusters <= 30:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=8)
    else:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "clusters_tsne.png"), dpi=300, bbox_inches="tight")
    print(f"Saved t-SNE visualization to {os.path.join(save_dir, 'clusters_tsne.png')}")
    plt.close()

    # PCA with more components for scree plot
    print("\nComputing PCA scree plot...")
    pca_full = PCA(n_components=min(50, embeddings_vis.shape[1]))
    pca_full.fit(embeddings_vis)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(
        range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_,
        "bo-",
    )
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum) + 1), cumsum, "ro-")
    plt.axhline(y=0.95, color="g", linestyle="--", label="95% variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Variance Explained")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "pca_variance.png"), dpi=300, bbox_inches="tight"
    )
    print(f"Saved PCA variance analysis to {os.path.join(save_dir, 'pca_variance.png')}")
    plt.close()


def load_data_from_hdf5(hdf5_path):
    """
    Load embeddings and metadata from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 embeddings file

    Returns:
        dict with:
            - encoder_embeddings: (N, T', D) numpy array
            - code_indices: (N, T') numpy array
            - texts: list of text descriptions
            - names: list of sample IDs
            - compound_verbs: list of compound verb labels (e.g., "walk-run-sit")
    """
    print(f"Loading data from HDF5: {hdf5_path}")

    encoder_embeddings = []
    code_indices = []
    texts = []
    names = []
    compound_verbs = []

    with h5py.File(hdf5_path, "r") as f:
        for sample_id in f.keys():
            sample_group = f[sample_id]

            encoder_embeddings.append(sample_group["encoder_embeddings"][:])
            code_indices.append(sample_group["code_indices"][:])
            texts.append(sample_group.attrs["text"])
            names.append(sample_id)
            compound_verbs.append(sample_group.attrs["compound_verb"])

    return {
        "encoder_embeddings": np.array(encoder_embeddings),
        "code_indices": np.array(code_indices),
        "texts": texts,
        "names": names,
        "compound_verbs": compound_verbs,
    }


def label_clusters_by_majority_verb(labels, compound_verbs):
    """
    Label each k-means cluster by the most frequent atomic verb within it.

    For each cluster:
    1. Collect all compound verb labels (e.g., "walk-run-sit")
    2. Break them into atomic verbs
    3. Find the most frequent atomic verb across all samples in that cluster

    Args:
        labels: (N,) k-means cluster assignments
        compound_verbs: list of compound verb strings (e.g., ["walk-run", "sit", ...])

    Returns:
        cluster_verb_labels: dict mapping cluster_id to most frequent atomic verb
        cluster_atomic_verb_counts: dict mapping cluster_id to Counter of atomic verbs
        cluster_compound_verb_counts: dict mapping cluster_id to Counter of compound verbs
    """
    if compound_verbs is None:
        return None, None, None

    n_clusters = len(np.unique(labels))
    cluster_atomic_verb_counts = defaultdict(Counter)
    cluster_compound_verb_counts = defaultdict(Counter)

    # Count verbs for each cluster
    for i, cluster_id in enumerate(labels):
        compound_verb = compound_verbs[i] if i < len(compound_verbs) else ""

        # Count the compound verb itself
        cluster_compound_verb_counts[cluster_id][compound_verb] += 1

        # Break into atomic verbs and count each one
        if compound_verb:
            atomic_verbs = compound_verb.split("-")
            for verb in atomic_verbs:
                cluster_atomic_verb_counts[cluster_id][verb] += 1

    # Find most frequent atomic verb for each cluster
    cluster_verb_labels = {}
    for cluster_id in range(n_clusters):
        atomic_counts = cluster_atomic_verb_counts[cluster_id]
        if atomic_counts:
            most_frequent_verb = atomic_counts.most_common(1)[0][0]
            cluster_verb_labels[cluster_id] = most_frequent_verb
        else:
            cluster_verb_labels[cluster_id] = "unknown"

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

        # Store cluster labels as a root-level dataset
        cluster_labels_ds = f.create_dataset("cluster_labels", data=labels)
        cluster_labels_ds.attrs["description"] = "K-means cluster assignments for each sample"

        # Store cluster verb labels mapping
        for cluster_id, verb_label in cluster_verb_labels.items():
            f.attrs[f"cluster_{cluster_id}_label"] = verb_label

        # Create a group for each sample
        for idx, name in enumerate(tqdm(data["names"], desc="Writing clustered HDF5")):
            sample_group = f.create_group(name)

            # Add original compound verb
            sample_group.attrs["compound_verb"] = data["compound_verbs"][idx]

            # Add cluster assignment and label
            sample_group.attrs["cluster_id"] = int(labels[idx])
            sample_group.attrs["cluster_label"] = cluster_verb_labels[labels[idx]]

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
    cluster_verb_labels=None, cluster_atomic_verb_counts=None, cluster_compound_verb_counts=None
):
    """Analyze what each K-means cluster represents."""

    print("\n" + "=" * 80)
    print("CLUSTER ANALYSIS (K-MEANS)")
    print("=" * 80)

    n_clusters = len(np.unique(labels))

    analysis_path = os.path.join(save_dir, "cluster_analysis.txt")
    analysis_file = open(analysis_path, "w")

    # Also save cluster labels
    if cluster_verb_labels is not None:
        cluster_labels_path = os.path.join(save_dir, "cluster_verb_labels.txt")
        with open(cluster_labels_path, "w") as f:
            for cluster_id in range(n_clusters):
                f.write(f"{cluster_id}\t{cluster_verb_labels[cluster_id]}\n")
        print(f"Saved cluster verb labels to {cluster_labels_path}")

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_texts = [texts[i] for i, m in enumerate(mask) if m]
        cluster_codes = code_indices[mask]

        # Get cluster label
        cluster_label = ""
        if cluster_verb_labels is not None:
            cluster_label = f" ({cluster_verb_labels[cluster_id]})"

        analysis = f"\n{'='*80}\n"
        analysis += (
            f"K-Means Cluster {cluster_id}{cluster_label} - {len(cluster_texts)} samples "
            f"({len(cluster_texts)/len(labels)*100:.1f}%)\n"
        )
        analysis += f"{'='*80}\n"

        # Sample text descriptions
        analysis += f"\nSample descriptions (first {top_k}):\n"
        for i, text in enumerate(cluster_texts[:top_k]):
            analysis += f"  {i+1}. {text}\n"

        # Most common discrete codes used
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
    parser.add_argument(
        "--data-dir",
        default="clustering/outputs",
        type=str,
        help="Directory containing encoder_embeddings.npy, code_indices.npy, texts.txt, names.txt",
    )
    parser.add_argument(
        "--save-dir",
        default="clustering/outputs",
        type=str,
        help="Directory to save clustering results (defaults to data-dir)",
    )
    parser.add_argument(
        "--n-clusters",
        default=20,
        type=int,
        help="Number of clusters for K-means on encoder embeddings",
    )
    parser.add_argument(
        "--hdf5-file",
        default=None,
        type=str,
        help="Path to HDF5 embeddings file (e.g., embeddings_train.h5). If provided, will use this instead of numpy files.",
    )
    parser.add_argument(
        "--elbow-method",
        action="store_true",
        help="Run elbow method to determine optimal number of clusters",
    )
    parser.add_argument(
        "--max-clusters",
        default=30,
        type=int,
        help="Maximum number of clusters to test in elbow method (default: 30)",
    )
    parser.add_argument(
        "--auto-k",
        action="store_true",
        help="Automatically use optimal k from elbow method (requires --elbow-method)",
    )
    parser.add_argument(
        "--auto-k-method",
        default="silhouette",
        choices=["silhouette", "derivative"],
        help="Method to choose optimal k: 'silhouette' (best silhouette score) or 'derivative' (elbow point)",
    )

    args = parser.parse_args()

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Load extracted data
    print("=" * 80)
    print("Loading extracted embeddings and metadata...")
    print("=" * 80)

    # Load from HDF5 or legacy numpy files
    if args.hdf5_file:
        # Load from HDF5
        data = load_data_from_hdf5(args.hdf5_file)
        encoder_embeddings = data["encoder_embeddings"]
        code_indices = data["code_indices"]
        texts = data["texts"]
        names = data["names"]
        compound_verbs = data["compound_verbs"]
    else:
        # Load from legacy numpy files
        encoder_embeddings = np.load(os.path.join(args.data_dir, "encoder_embeddings.npy"))
        code_indices = np.load(os.path.join(args.data_dir, "code_indices.npy"))

        with open(os.path.join(args.data_dir, "texts.txt"), "r") as f:
            texts = f.read().strip().split("\n")
        with open(os.path.join(args.data_dir, "names.txt"), "r") as f:
            names = f.read().strip().split("\n")

        # No compound verbs available from legacy format
        compound_verbs = None

    print(f"  Encoder embeddings: {encoder_embeddings.shape}")
    print(f"  Code indices: {code_indices.shape}")
    print(f"  Number of texts: {len(texts)}")
    print(f"  Number of samples: {len(names)}")

    # Run elbow method if requested
    optimal_k = args.n_clusters
    if args.elbow_method:
        print("\n" + "=" * 80)
        print("Running Elbow Method...")
        print("=" * 80)
        _, _, _, suggested_k, best_silhouette_k = elbow_method(
            encoder_embeddings,
            max_clusters=args.max_clusters,
            aggregate="mean",
            pca_dim=50,
            save_dir=args.save_dir,
        )

        # Auto-select k if requested
        if args.auto_k:
            if args.auto_k_method == "silhouette" and best_silhouette_k is not None:
                optimal_k = best_silhouette_k
                print(f"\n>>> Auto-selecting k={optimal_k} based on silhouette score")
            elif args.auto_k_method == "derivative":
                optimal_k = suggested_k
                print(f"\n>>> Auto-selecting k={optimal_k} based on elbow point")
            else:
                print(f"\n>>> Could not auto-select k, using default k={optimal_k}")

    # K-means clustering on encoder embeddings
    print("\n" + "=" * 80)
    print(f"K-Means Clustering embeddings with k={optimal_k}...")
    print("=" * 80)

    labels, kmeans, embeddings_processed, _ = cluster_embeddings(
        encoder_embeddings,
        n_clusters=optimal_k,
        aggregate="mean",
        pca_dim=50,
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
    visualize_clusters(embeddings_processed, labels, texts, args.save_dir, cluster_verb_labels=cluster_verb_labels)

    # Analyze K-means clusters
    print("\n" + "=" * 80)
    print("Analyzing K-means clusters...")
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
    )

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
    if args.elbow_method:
        print("  - elbow_method.png: elbow method plot with inertia and silhouette scores")
    print("  - cluster_labels.npy: K-means cluster assignments (N,)")
    if cluster_verb_labels:
        print("  - cluster_verb_labels.txt: atomic verb label for each cluster")
    print("  - embeddings_processed.npy: processed embeddings used for clustering")
    print("  - clusters_pca.png: PCA visualization with atomic verb labels")
    print("  - clusters_tsne.png: t-SNE visualization with atomic verb labels")
    print("  - pca_variance.png: PCA variance analysis")
    print("  - cluster_analysis.txt: detailed cluster analysis (atomic + compound verb distributions)")
    if args.hdf5_file:
        print("  - embeddings_clustered.h5: HDF5 file with cluster assignments and atomic verb labels")


if __name__ == "__main__":
    main()


