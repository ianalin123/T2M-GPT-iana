import os
import sys
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Ensure project root is on the path so we can import project modules if needed
# Since we're now in clustering/, go up one level to get to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from clustering.verb_clustering_utils import ( 
    cluster_by_primary_verb,
    cluster_by_frequent_verbs,
    cluster_verbs_semantic,
    cluster_by_verb_category,
    analyze_verb_label_statistics,
)


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


def visualize_clusters(embeddings, labels, _texts, save_dir, max_samples=100000):
    """Create PCA visualizations of clustered embeddings."""

    # Subsample if too many
    if len(embeddings) > max_samples:
        print(f"\nSubsampling {max_samples} points for visualization...")
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings_vis = embeddings[indices]
        labels_vis = labels[indices]
    else:
        embeddings_vis = embeddings
        labels_vis = labels

    # PCA - Fast and interpretable!
    print("\nComputing PCA for visualization...")
    pca_viz = PCA(n_components=2)
    embeddings_pca = pca_viz.fit_transform(embeddings_vis)

    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(
        embeddings_pca[:, 0],
        embeddings_pca[:, 1],
        c=labels_vis,
        cmap="tab20",
        alpha=0.6,
        s=10,
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title(
        "PCA Visualization of Latent Embeddings\n"
        f"(PC1: {pca_viz.explained_variance_ratio_[0]:.1%}, "
        f"PC2: {pca_viz.explained_variance_ratio_[1]:.1%}, "
        f"Total: {pca_viz.explained_variance_ratio_.sum():.1%})"
    )
    plt.xlabel(f"PC1 ({pca_viz.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca_viz.explained_variance_ratio_[1]:.1%} variance)")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "clusters_pca.png"), dpi=300, bbox_inches="tight")
    print(f"Saved PCA visualization to {os.path.join(save_dir, 'clusters_pca.png')}")
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


def load_verb_labels(verbs_file="dataset/HumanML3D/verbs.txt"):
    """
    Load verb labels from extract_verbs.py output.

    Returns:
        dict: mapping from filename (without .txt) to verb label string
    """
    verb_labels = {}
    if not os.path.exists(verbs_file):
        print(f"Warning: Verb labels file not found at {verbs_file}")
        print("         Verb-based clustering will be skipped.")
        return None

    with open(verbs_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                filename, verb_label = parts
                verb_labels[filename] = verb_label

    print(f"Loaded {len(verb_labels)} verb labels from {verbs_file}")
    return verb_labels


# def cluster_by_verbs(names, verb_labels, method="all", num_clusters=20):
#     """
#     Cluster samples by their verb labels using various strategies.

#     Args:
#         names: list of sample names (may include timestamps like name_start_end)
#         verb_labels: dict mapping filename to verb label
#         method: clustering method - 'all', 'primary', 'frequent', 'semantic', 'category'
#         num_clusters: target number of clusters (used for 'frequent' and 'semantic')

#     Returns:
#         verb_cluster_labels: (N,) array of cluster IDs based on verb labels
#         verb_label_to_id: dict mapping label/category to cluster ID
#     """
#     if verb_labels is None:
#         return None, None

#     print(f"\nVerb clustering method: {method}")

#     if method == "all":
#         # Original method: one cluster per unique verb combination
#         verb_cluster_labels = []
#         verb_label_to_id = {}
#         next_id = 0

#         for name in names:
#             base_name = name.split("_")[0] if "_" in name else name
#             verb_label = verb_labels.get(base_name, "unknown")

#             if verb_label not in verb_label_to_id:
#                 verb_label_to_id[verb_label] = next_id
#                 next_id += 1

#             verb_cluster_labels.append(verb_label_to_id[verb_label])

#         verb_cluster_labels = np.array(verb_cluster_labels)

#     elif method == "primary":
#         verb_cluster_labels, verb_label_to_id = cluster_by_primary_verb(
#             names, verb_labels
#         )

#     elif method == "frequent":
#         verb_cluster_labels, verb_label_to_id = cluster_by_frequent_verbs(
#             names, verb_labels, num_clusters=num_clusters
#         )

#     elif method == "semantic":
#         verb_cluster_labels, verb_label_to_id = cluster_verbs_semantic(
#             names, verb_labels, num_clusters=num_clusters
#         )

#     elif method == "category":
#         verb_cluster_labels, verb_label_to_id = cluster_by_verb_category(
#             names, verb_labels
#         )

#     else:
#         raise ValueError(f"Unknown verb clustering method: {method}")

#     # Print verb cluster distribution
#     print("\nVerb-based cluster distribution:")
#     unique_labels, counts = np.unique(verb_cluster_labels, return_counts=True)

#     if method in {"all", "primary", "frequent"}:
#         id_to_label = {v: k for k, v in verb_label_to_id.items()}
#         for label_id, count in sorted(
#             zip(unique_labels, counts), key=lambda x: -x[1]
#         )[:20]:
#             verb_label = id_to_label.get(label_id, "unknown")
#             print(
#                 f"  Verb cluster {label_id:2d} ({verb_label:30s}): "
#                 f"{count:5d} samples ({count/len(verb_cluster_labels)*100:5.1f}%)"
#             )
#     else:
#         for label_id, count in sorted(zip(unique_labels, counts), key=lambda x: -x[1]):
#             print(
#                 f"  Verb cluster {label_id:2d}: "
#                 f"{count:5d} samples ({count/len(verb_cluster_labels)*100:5.1f}%)"
#             )

#     return verb_cluster_labels, verb_label_to_id


def label_clusters_by_atomic_verbs(labels, names, verb_labels):
    """
    Label each k-means cluster by the most frequent atomic verb within it.
    Splits compound verbs like "run-jump" into atomic verbs ["run", "jump"].

    Args:
        labels: (N,) k-means cluster assignments
        names: list of sample names
        verb_labels: dict mapping filename to verb label

    Returns:
        cluster_verb_labels: dict mapping cluster_id to most frequent atomic verb
        cluster_atomic_verb_counts: dict mapping cluster_id to Counter of atomic verbs
    """
    if verb_labels is None:
        return None, None

    n_clusters = len(np.unique(labels))
    cluster_atomic_verb_counts = defaultdict(lambda: defaultdict(int))

    # Count atomic verbs for each cluster
    for i, cluster_id in enumerate(labels):
        name = names[i]
        base_name = name.split("_")[0] if "_" in name else name
        verb_label = verb_labels.get(base_name, "unknown")

        # Split compound verbs into atomic verbs
        atomic_verbs = verb_label.replace("-", " ").split()

        for atomic_verb in atomic_verbs:
            cluster_atomic_verb_counts[cluster_id][atomic_verb] += 1

    # Find most frequent atomic verb for each cluster
    cluster_verb_labels = {}
    for cluster_id in range(n_clusters):
        verb_counts = cluster_atomic_verb_counts[cluster_id]
        if verb_counts:
            most_frequent_verb = max(verb_counts.items(), key=lambda x: x[1])[0]
            cluster_verb_labels[cluster_id] = most_frequent_verb
        else:
            cluster_verb_labels[cluster_id] = "unknown"

    return cluster_verb_labels, cluster_atomic_verb_counts


def analyze_clusters(
    labels, texts, code_indices, save_dir, top_k=10, verb_labels=None, names=None
):
    """Analyze what each K-means cluster represents."""

    print("\n" + "=" * 80)
    print("CLUSTER ANALYSIS (K-MEANS)")
    print("=" * 80)

    n_clusters = len(np.unique(labels))

    # Label clusters by atomic verbs
    cluster_verb_labels, cluster_atomic_verb_counts = label_clusters_by_atomic_verbs(
        labels, names, verb_labels
    )

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
            verb_counts = cluster_atomic_verb_counts[cluster_id]
            analysis += "\nAtomic verb distribution in this cluster:\n"
            for verb, count in sorted(verb_counts.items(), key=lambda x: -x[1])[:15]:
                analysis += (
                    f"  {verb:20s}: {count:4d} samples "
                    f"({count/len(cluster_texts)*100:.1f}%)\n"
                )

        print(analysis)
        analysis_file.write(analysis)

    analysis_file.close()
    print(f"\nSaved full analysis to {analysis_path}")


def analyze_verb_clusters(
    verb_cluster_labels, verb_label_to_id, texts, code_indices, names, save_dir, top_k=10
):
    """Analyze verb-based clusters."""

    if verb_cluster_labels is None:
        return

    print("\n" + "=" * 80)
    print("VERB-BASED CLUSTER ANALYSIS")
    print("=" * 80)

    n_clusters = len(np.unique(verb_cluster_labels))
    id_to_label = {v: k for k, v in verb_label_to_id.items()}

    analysis_path = os.path.join(save_dir, "verb_cluster_analysis.txt")
    analysis_file = open(analysis_path, "w")

    for cluster_id in range(n_clusters):
        mask = verb_cluster_labels == cluster_id
        cluster_texts = [texts[i] for i, m in enumerate(mask) if m]
        cluster_codes = code_indices[mask]
        verb_label = id_to_label[cluster_id]

        analysis = f"\n{'='*80}\n"
        analysis += (
            f"Verb Cluster {cluster_id} ({verb_label}) - {len(cluster_texts)} samples "
            f"({len(cluster_texts)/len(verb_cluster_labels)*100:.1f}%)\n"
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

        print(analysis)
        analysis_file.write(analysis)

    analysis_file.close()
    print(f"\nSaved verb cluster analysis to {analysis_path}")


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
    # parser.add_argument(
    #     "--verb-cluster-method",
    #     default="frequent",
    #     choices=["all", "primary", "frequent", "semantic", "category"],
    #     help=(
    #         "Method for reducing verb labels to clusters:\n"
    #         "  all: one cluster per unique verb combination (413+ clusters)\n"
    #         "  primary: use only first verb from each combination\n"
    #         "  frequent: keep top-K most frequent combinations, group rest as 'other'\n"
    #         "  semantic: cluster using word embeddings (semantically similar verbs)\n"
    #         "  category: manual semantic categories (locomotion, upper_body, etc.)"
    #     ),
    # )
    # parser.add_argument(
    #     "--verb-clusters",
    #     default=None,
    #     type=int,
    #     help="Number of verb clusters (for frequent/semantic methods). Defaults to --n-clusters",
    # )
    parser.add_argument(
        "--verbs-file",
        default="dataset/HumanML3D/verbs.txt",
        type=str,
        help="Path to verbs.txt produced by extract_verbs.py",
    )
    parser.add_argument(
        "--analyze-verbs",
        action="store_true",
        help="Print detailed statistics about verb labels before clustering",
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

    if args.verb_clusters is None:
        args.verb_clusters = args.n_clusters

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Load extracted data
    print("=" * 80)
    print("Loading extracted embeddings and metadata...")
    print("=" * 80)

    encoder_embeddings = np.load(os.path.join(args.data_dir, "encoder_embeddings.npy"))
    code_indices = np.load(os.path.join(args.data_dir, "code_indices.npy"))

    with open(os.path.join(args.data_dir, "texts.txt"), "r") as f:
        texts = f.read().strip().split("\n")
    with open(os.path.join(args.data_dir, "names.txt"), "r") as f:
        names = f.read().strip().split("\n")

    print(f"  Encoder embeddings: {encoder_embeddings.shape}")
    print(f"  Code indices: {code_indices.shape}")
    print(f"  Number of texts: {len(texts)}")
    print(f"  Number of samples: {len(names)}")

    # Load verb labels
    print("\n" + "=" * 80)
    print("Loading verb labels...")
    print("=" * 80)
    verb_labels = load_verb_labels(args.verbs_file)

    # Analyze verb labels if requested
    if args.analyze_verbs and verb_labels is not None:
        analyze_verb_label_statistics(verb_labels)

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

    # # Verb-based clustering
    # print("\n" + "=" * 80)
    # print("Verb-based Clustering...")
    # print("=" * 80)
    # verb_cluster_labels, verb_label_to_id = cluster_by_verbs(
    #     names,
    #     verb_labels,
    #     method=args.verb_cluster_method,
    #     num_clusters=args.verb_clusters,
    # )

    if verb_cluster_labels is not None:
        np.save(
            os.path.join(args.save_dir, "verb_cluster_labels.npy"),
            verb_cluster_labels,
        )
        # Save verb label mapping
        mapping_path = os.path.join(args.save_dir, "verb_label_mapping.txt")
        with open(mapping_path, "w") as f:
            for verb_label, cluster_id in sorted(
                verb_label_to_id.items(), key=lambda x: x[1]
            ):
                f.write(f"{cluster_id} {verb_label}\n")
        print(f"Saved verb label mapping to {mapping_path}")

    # Visualize clusters
    print("\n" + "=" * 80)
    print("Visualizing clusters...")
    print("=" * 80)
    visualize_clusters(embeddings_processed, labels, texts, args.save_dir)

    # Analyze K-means clusters
    print("\n" + "=" * 80)
    print("Analyzing K-means clusters...")
    print("=" * 80)
    analyze_clusters(
        labels,
        texts,
        code_indices,
        args.save_dir,
        verb_labels=verb_labels,
        names=names,
    )

    # # Analyze verb clusters
    # if verb_cluster_labels is not None:
    #     print("\n" + "=" * 80)
    #     print("Analyzing verb-based clusters...")
    #     print("=" * 80)
    #     analyze_verb_clusters(
    #         verb_cluster_labels,
    #         verb_label_to_id,
    #         texts,
    #         code_indices,
    #         names,
    #         args.save_dir,
    #     )

    print("\n" + "=" * 80)
    print(f"✓ Done! All results saved to {args.save_dir}/")
    print("=" * 80)
    print("\nGenerated files:")
    if args.elbow_method:
        print("  - elbow_method.png: elbow method plot with inertia and silhouette scores")
    print("  - cluster_labels.npy: K-means cluster assignments (N,)")
    print("  - cluster_verb_labels.txt: most frequent atomic verb label for each cluster")
    # if verb_cluster_labels is not None:
    #     print("  - verb_cluster_labels.npy: verb-based cluster assignments (N,)")
    #     print("  - verb_label_mapping.txt: mapping of verb labels/categories to cluster IDs")
    #     print("  - verb_cluster_analysis.txt: detailed verb-based cluster analysis")
    print("  - embeddings_processed.npy: processed embeddings used for clustering")
    print("  - clusters_pca.png: PCA visualization")
    print("  - pca_variance.png: PCA variance analysis")
    print("  - cluster_analysis.txt: detailed K-means cluster analysis")


if __name__ == "__main__":
    main()


