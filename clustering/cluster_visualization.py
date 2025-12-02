import os
from collections import defaultdict

import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA


def visualize_clusters(
    embeddings,
    labels,
    _texts,
    save_dir,
    max_samples=100000,
    cluster_verb_labels=None,
    dim_reduction_method=None,
):
    """
    Visualize clusters using PCA, optional UMAP (if embeddings are already reduced),
    and t-SNE. This is extracted from run_clustering.py so it can be reused
    independently of the clustering step.
    """

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
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "axes.linewidth": 1.0,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 300,
        }
    )

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

    # --------------------------------------------------------
    # Strong, perceptually uniform color palette (HUSL)
    # Works great for <= 30 clusters
    # --------------------------------------------------------
    non_noise_labels = [cid for cid in unique_labels if cid != -1]
    colors = sns.color_palette("husl", len(non_noise_labels) or 1)
    cluster_id_to_color_idx = {cid: i for i, cid in enumerate(non_noise_labels)}

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
                label=cluster_label,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        # Legend outside figure
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            title="Clusters",
            title_fontsize=13,
        )

        ax.margins(0.05)
        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", metadata={"Creator": None})
        plt.savefig(
            save_path.replace(".png", ".pdf"),
            dpi=300,
            bbox_inches="tight",
            metadata={"Creator": None},
        )
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
        verbose=1,
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


def analyze_clusters(
    labels,
    texts,
    code_indices,
    save_dir,
    top_k=10,
    compound_verbs=None,
    cluster_verb_labels=None,
    cluster_atomic_verb_counts=None,
    cluster_compound_verb_counts=None,
    algorithm="kmeans",
    names=None,
):
    """
    Analyze what each cluster represents and write a detailed report to
    cluster_analysis.txt. Extracted from run_clustering.py so it can be reused
    independently, e.g. after post-processing labels.
    """

    # Map algorithm to display name
    algo_display = {
        "kmeans": "K-Means",
        "gmm": "Gaussian Mixture Model",
        "hdbscan": "HDBSCAN",
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
                sample_name = (
                    names[sample_idx]
                    if names is not None and sample_idx < len(names)
                    else f"sample_{sample_idx}"
                )
                compound_verb = (
                    compound_verbs[sample_idx]
                    if compound_verbs is not None and sample_idx < len(compound_verbs)
                    else "N/A"
                )
                f.write(f"{idx}. Sample: {sample_name}\n")
                f.write(f"   Text: {text}\n")
                f.write(f"   Compound Verb: {compound_verb}\n")
                f.write("\n")

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
                analysis += f"  {verb:20s}: {count:4d} occurrences\n"

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


def _load_optional_npy(path):
    if os.path.exists(path):
        return np.load(path)
    return None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize and analyze clusters using precomputed embeddings and labels.\n"
            "Useful after running run_clustering.py and merge_semantic_clusters.py."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="clustering/outputs",
        help="Directory containing embeddings_processed.npy, labels, texts, etc.",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default="cluster_labels_denoised.npy",
        help="NumPy file with cluster labels to analyze/visualize "
        "(e.g., cluster_labels.npy, cluster_labels_merged.npy, "
        "cluster_labels_denoised.npy).",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="hdbscan",
        choices=["kmeans", "gmm", "hdbscan"],
        help="Clustering algorithm used to produce the labels (affects reporting).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100000,
        help="Maximum number of points to use for visualization.",
    )
    parser.add_argument(
        "--dim-reduction-method",
        type=str,
        default="umap",
        choices=["pca", "umap", "none"],
        help="How to interpret embeddings for 2D visualization: "
        "'umap' assumes they are already UMAP-reduced, 'pca' recomputes PCA, "
        "'none' skips UMAP-specific plots.",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    labels_path = os.path.join(data_dir, args.labels_file)
    emb_path = os.path.join(data_dir, "embeddings_processed.npy")

    print("=" * 80)
    print(f"Loading embeddings and labels from {data_dir} ...")
    print("=" * 80)

    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"embeddings_processed.npy not found in {data_dir}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"{args.labels_file} not found in {data_dir}")

    embeddings = np.load(emb_path)
    labels = np.load(labels_path)

    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Mismatch between embeddings ({embeddings.shape[0]}) and labels "
            f"({labels.shape[0]})."
        )

    texts_path = os.path.join(data_dir, "texts.txt")
    texts = []
    if os.path.exists(texts_path):
        with open(texts_path, "r") as f:
            texts = f.read().strip().split("\n")
        if len(texts) != embeddings.shape[0]:
            print(
                f"Warning: {len(texts)} texts but {embeddings.shape[0]} embeddings. "
                "Truncating to minimum length."
            )
            n = min(len(texts), embeddings.shape[0])
            texts = texts[:n]
            embeddings = embeddings[:n]
            labels = labels[:n]
    else:
        texts = [""] * embeddings.shape[0]

    code_indices = _load_optional_npy(os.path.join(data_dir, "code_indices.npy"))
    if code_indices is not None and code_indices.shape[0] != embeddings.shape[0]:
        print(
            f"Warning: code_indices shape {code_indices.shape} does not match "
            f"embeddings {embeddings.shape[0]}; ignoring code_indices."
        )
        code_indices = None

    # Optional: names and compound_verbs (used mainly for nicer reporting)
    names = None
    names_path = os.path.join(data_dir, "names.txt")
    if os.path.exists(names_path):
        with open(names_path, "r") as f:
            names = f.read().strip().split("\n")
        if len(names) != embeddings.shape[0]:
            print(
                f"Warning: {len(names)} names but {embeddings.shape[0]} embeddings; "
                "ignoring names."
            )
            names = None

    compound_verbs = None  # not easily reconstructable here; optional

    # Load cluster_verb_labels if available (from a previous analysis run)
    cluster_verb_labels = {}
    cvl_path = os.path.join(data_dir, "cluster_verb_labels.txt")
    if os.path.exists(cvl_path):
        with open(cvl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cid_str, verb = line.split("\t", 1)
                try:
                    cid = int(cid_str)
                except ValueError:
                    continue
                cluster_verb_labels[cid] = verb
        print(f"Loaded cluster verb labels from {cvl_path}")
    else:
        cluster_verb_labels = None

    # Visualization
    visualize_clusters(
        embeddings,
        labels,
        texts,
        data_dir,
        max_samples=args.max_samples,
        cluster_verb_labels=cluster_verb_labels,
        dim_reduction_method=None if args.dim_reduction_method == "none" else args.dim_reduction_method,
    )

    # Analysis
    analyze_clusters(
        labels,
        texts,
        code_indices,
        data_dir,
        compound_verbs=compound_verbs,
        cluster_verb_labels=cluster_verb_labels,
        cluster_atomic_verb_counts=None,
        cluster_compound_verb_counts=None,
        algorithm=args.algorithm,
        names=names,
    )


if __name__ == "__main__":
    main()

