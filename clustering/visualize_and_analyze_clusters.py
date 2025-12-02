#!/usr/bin/env python3
"""
Visualize and analyze clustering results from HDF5 files.

This script loads clustered embeddings from HDF5 and generates:
- Visualization plots (PCA, t-SNE, UMAP)
- Detailed cluster analysis reports

Usage:
    python clustering/visualize_and_analyze_clusters.py --hdf5-file clustering/outputs/embeddings_clustered.h5
    python clustering/visualize_and_analyze_clusters.py --hdf5-file clustering/outputs/embeddings_clustered_denoised.h5
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Check for optional UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def load_data_from_hdf5(hdf5_path):
    """
    Load clustered embeddings and metadata from HDF5 file.

    Returns:
        dict with:
            - embeddings: (N, T', D) numpy array of encoder embeddings
            - labels: (N,) cluster assignments
            - texts: list of text descriptions
            - names: list of sample IDs
            - compound_verbs: list of compound verb labels
            - cluster_verb_labels: dict mapping cluster_id to atomic verb
    """
    print(f"Loading data from HDF5: {hdf5_path}")
    
    embeddings = []
    labels = []
    texts = []
    names = []
    compound_verbs = []
    cluster_verb_labels = {}

    with h5py.File(hdf5_path, "r") as f:
        # Load cluster verb labels from root attributes
        for attr_name in f.attrs.keys():
            if attr_name.startswith("cluster_") and attr_name.endswith("_label"):
                # Extract cluster ID from attribute name
                cid_str = attr_name.replace("cluster_", "").replace("_label", "")
                try:
                    cid = int(cid_str)
                    cluster_verb_labels[cid] = f.attrs[attr_name]
                except ValueError:
                    continue

        # Load per-sample data
        for sample_id in tqdm(f.keys(), desc="Loading samples"):
            sample_group = f[sample_id]

            embeddings.append(sample_group["encoder_embeddings"][:])
            labels.append(sample_group.attrs["cluster_id"])
            texts.append(sample_group.attrs["text"])
            names.append(sample_id)
            compound_verbs.append(sample_group.attrs.get("compound_verb", ""))

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    print(f"  Loaded {len(names)} samples")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Found {len(cluster_verb_labels)} cluster verb labels")

    return {
        "embeddings": embeddings,
        "labels": labels,
        "texts": texts,
        "names": names,
        "compound_verbs": compound_verbs,
        "cluster_verb_labels": cluster_verb_labels,
    }


def aggregate_embeddings(embeddings, method="mean"):
    """
    Aggregate embeddings over time dimension.
    
    Args:
        embeddings: (N, T, D) array
        method: 'mean' or 'max'
    
    Returns:
        (N, D) aggregated embeddings
    """
    if embeddings.ndim == 2:
        # Already aggregated
        return embeddings
    
    if method == "mean":
        return embeddings.mean(axis=1)
    elif method == "max":
        return embeddings.max(axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def visualize_clusters(
    embeddings,
    labels,
    save_dir,
    max_samples=100000,
    cluster_verb_labels=None,
):
    """
    Create visualization plots for clusters using PCA, t-SNE, and optionally UMAP.
    
    Args:
        embeddings: (N, D) aggregated embeddings
        labels: (N,) cluster assignments
        save_dir: directory to save plots
        max_samples: maximum samples to visualize
        cluster_verb_labels: dict mapping cluster_id to verb label
    """
    print("\n" + "=" * 80)
    print("Visualizing clusters...")
    print("=" * 80)

    # Global plotting style
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

    # Subsample if needed
    if len(embeddings) > max_samples:
        print(f"Subsampling {max_samples} points for visualization...")
        idx = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings_vis = embeddings[idx]
        labels_vis = labels[idx]
    else:
        embeddings_vis = embeddings
        labels_vis = labels

    unique_labels = np.unique(labels_vis)
    
    # Color palette
    non_noise_labels = [cid for cid in unique_labels if cid != -1]
    colors = sns.color_palette("husl", len(non_noise_labels))
    cluster_id_to_color_idx = {cid: i for i, cid in enumerate(non_noise_labels)}

    def plot_clusters(embeddings_2d, xlabel, ylabel, title, save_path):
        fig, ax = plt.subplots(figsize=(10, 8))

        for cid in unique_labels:
            mask = labels_vis == cid

            if cid == -1:
                cluster_label = "Noise"
                point_color = (0.75, 0.75, 0.75)
            else:
                if cluster_verb_labels and cid in cluster_verb_labels:
                    v = cluster_verb_labels[cid]
                    cluster_label = v if v else f"Cluster {cid}"
                else:
                    cluster_label = f"Cluster {cid}"

                color_idx = cluster_id_to_color_idx.get(cid, 0)
                point_color = colors[color_idx]

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
        ax.set_title(title, pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

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

    # PCA 2D
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

    # t-SNE
    print("\nComputing t-SNE for visualization...")
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

    # Temporarily update labels_vis for t-SNE plot
    labels_vis_backup = labels_vis
    labels_vis = labels_tsne
    
    plot_clusters(
        emb_tsne,
        "t-SNE-1",
        "t-SNE-2",
        "t-SNE (2D Projection)",
        os.path.join(save_dir, "clusters_tsne.png"),
    )
    
    labels_vis = labels_vis_backup

    # UMAP (if available)
    if UMAP_AVAILABLE:
        print("\nComputing UMAP for visualization...")
        reducer = umap.UMAP(n_components=2, random_state=42, verbose=True)
        emb_umap = reducer.fit_transform(embeddings_vis)

        plot_clusters(
            emb_umap,
            "UMAP-1",
            "UMAP-2",
            "UMAP (2D Projection)",
            os.path.join(save_dir, "clusters_umap.png"),
        )
    else:
        print("\nUMAP not available. Skipping UMAP visualization.")


def analyze_clusters(
    labels,
    texts,
    save_dir,
    top_k=10,
    compound_verbs=None,
    cluster_verb_labels=None,
    names=None,
):
    """
    Analyze what each cluster represents and save detailed report.
    
    Args:
        labels: (N,) cluster assignments
        texts: list of text descriptions
        save_dir: directory to save analysis
        top_k: number of sample descriptions to show per cluster
        compound_verbs: list of compound verb labels
        cluster_verb_labels: dict mapping cluster_id to atomic verb
        names: list of sample IDs
    """
    print("\n" + "=" * 80)
    print("CLUSTER ANALYSIS")
    print("=" * 80)

    analysis_path = os.path.join(save_dir, "cluster_analysis.txt")
    analysis_file = open(analysis_path, "w")

    # Get unique cluster IDs (excluding noise)
    cluster_ids = sorted([c for c in np.unique(labels) if c != -1])
    
    # Handle noise cluster separately
    if -1 in labels:
        noise_mask = labels == -1
        noise_texts = [texts[i] for i, m in enumerate(noise_mask) if m]
        noise_indices = [i for i, m in enumerate(noise_mask) if m]
        
        # Save noise samples to file
        noise_samples_path = os.path.join(save_dir, "noise_samples.txt")
        with open(noise_samples_path, "w") as f:
            f.write(f"Noise Samples - {len(noise_texts)} samples\n")
            f.write(f"{'='*80}\n")
            f.write("These samples were not assigned to any cluster.\n")
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
            f"Noise Points - {len(noise_texts)} samples "
            f"({len(noise_texts)/len(labels)*100:.1f}%)\n"
        )
        analysis += f"{'='*80}\n"
        analysis += "\nThese samples were not assigned to any cluster.\n"
        analysis_file.write(analysis)
        print(analysis)

    # Compute cluster statistics
    cluster_atomic_verb_counts = defaultdict(Counter)
    cluster_compound_verb_counts = defaultdict(Counter)

    if compound_verbs:
        for i, cluster_id in enumerate(labels):
            if cluster_id == -1:
                continue
            
            compound_verb = compound_verbs[i] if i < len(compound_verbs) else ""
            cluster_compound_verb_counts[cluster_id][compound_verb] += 1

            if compound_verb:
                atomic_verbs = compound_verb.split("-")
                for verb in atomic_verbs:
                    cluster_atomic_verb_counts[cluster_id][verb] += 1

    # Analyze each cluster
    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        cluster_texts = [texts[i] for i, m in enumerate(mask) if m]

        # Get cluster label
        cluster_label = ""
        if cluster_verb_labels is not None and cluster_id in cluster_verb_labels:
            cluster_label = f" ({cluster_verb_labels[cluster_id]})"

        analysis = f"\n{'='*80}\n"
        analysis += (
            f"Cluster {cluster_id}{cluster_label} - {len(cluster_texts)} samples "
            f"({len(cluster_texts)/len(labels)*100:.1f}%)\n"
        )
        analysis += f"{'='*80}\n"

        # Sample text descriptions
        analysis += f"\nSample descriptions (first {top_k}):\n"
        for i, text in enumerate(cluster_texts[:top_k]):
            analysis += f"  {i+1}. {text}\n"

        # Atomic verb distribution in this cluster
        if cluster_atomic_verb_counts:
            atomic_counts = cluster_atomic_verb_counts[cluster_id]
            analysis += "\nAtomic verb distribution in this cluster:\n"
            for verb, count in atomic_counts.most_common(15):
                analysis += (
                    f"  {verb:20s}: {count:4d} occurrences\n"
                )

        # Compound verb distribution in this cluster
        if cluster_compound_verb_counts:
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize and analyze clustering results from HDF5 files"
    )
    parser.add_argument(
        "--hdf5-file",
        type=str,
        required=True,
        help="Path to clustered HDF5 file (e.g., embeddings_clustered.h5 or embeddings_clustered_denoised.h5)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save visualizations and analysis (default: same as HDF5 file directory)",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Method to aggregate embeddings over time dimension (default: mean)",
    )
    parser.add_argument(
        "--max-vis-samples",
        type=int,
        default=100000,
        help="Maximum samples to use for visualization (default: 100000)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of sample descriptions to show per cluster (default: 10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine save directory
    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.hdf5_file)
    
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 80)
    print("Loading clustered data from HDF5...")
    print("=" * 80)

    # Load data
    data = load_data_from_hdf5(args.hdf5_file)

    # Aggregate embeddings
    print(f"\nAggregating embeddings using '{args.aggregate}' pooling...")
    embeddings_agg = aggregate_embeddings(data["embeddings"], method=args.aggregate)
    print(f"Aggregated embeddings shape: {embeddings_agg.shape}")

    # Visualize clusters
    visualize_clusters(
        embeddings_agg,
        data["labels"],
        args.save_dir,
        max_samples=args.max_vis_samples,
        cluster_verb_labels=data["cluster_verb_labels"],
    )

    # Analyze clusters
    analyze_clusters(
        data["labels"],
        data["texts"],
        args.save_dir,
        top_k=args.top_k,
        compound_verbs=data["compound_verbs"],
        cluster_verb_labels=data["cluster_verb_labels"],
        names=data["names"],
    )

    print("\n" + "=" * 80)
    print(f"✓ Done! All results saved to {args.save_dir}/")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - clusters_pca.png/pdf: PCA visualization")
    print("  - clusters_tsne.png/pdf: t-SNE visualization")
    if UMAP_AVAILABLE:
        print("  - clusters_umap.png/pdf: UMAP visualization")
    print("  - cluster_analysis.txt: detailed cluster analysis")
    if -1 in data["labels"]:
        print("  - noise_samples.txt: list of noise samples")


if __name__ == "__main__":
    main()

