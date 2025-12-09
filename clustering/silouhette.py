#!/usr/bin/env python3
"""
Compute silhouette score for FINAL semantic clusters (Option A).

- Loads embeddings from:    <data_dir>/embeddings_processed.npy
- Loads labels from (first that exists, in this order):
    1) <data_dir>/cluster_labels_denoised.npy
    2) <data_dir>/cluster_labels_merged.npy
    3) <data_dir>/cluster_labels.npy   (fallback)

- Excludes noise points (label == -1) from the silhouette computation.
- Uses *all* remaining points (no subsampling).
- Writes results to:        <data_dir>/semantic_cluster_metrics.txt

Run e.g.:
    python clustering/silouhette.py \
        --data-dir clustering/outputs_continuous --algorithm hdbscan
"""

import argparse
import os
from typing import Tuple, Optional

import numpy as np
from sklearn.metrics import silhouette_score


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute silhouette score for final semantic clusters "
            "(merged + optionally denoised), using embeddings_processed.npy."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="clustering/outputs",
        help=(
            "Directory containing embeddings_processed.npy and final labels "
            "(cluster_labels_denoised.npy or cluster_labels_merged.npy)."
        ),
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="hdbscan",
        help=(
            "Name of the clustering algorithm (for logging only), "
            "e.g., 'hdbscan', 'kmeans', 'gmm'."
        ),
    )
    return parser.parse_args()


def load_embeddings(data_dir: str) -> np.ndarray:
    emb_path = os.path.join(data_dir, "embeddings_processed.npy")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"embeddings_processed.npy not found in {data_dir}")
    embeddings = np.load(emb_path)
    print(f"Loaded embeddings from {emb_path}: {embeddings.shape}")

    # If 3D (N, T, D), aggregate over time
    if embeddings.ndim == 3:
        print("Embeddings are 3D (N, T, D); aggregating with mean over time.")
        embeddings = embeddings.mean(axis=1)
        print(f"Aggregated embeddings shape: {embeddings.shape}")
    elif embeddings.ndim != 2:
        raise ValueError(
            f"Expected embeddings of shape (N, D) or (N, T, D), "
            f"got shape {embeddings.shape}"
        )

    return embeddings


def load_final_labels(data_dir: str) -> Tuple[np.ndarray, str]:
    """
    Try, in order:
        cluster_labels_denoised.npy
        cluster_labels_merged.npy
        cluster_labels.npy
    Returns labels and the filename actually used.
    """
    candidates = [
        "cluster_labels_denoised.npy",
        "cluster_labels_merged.npy",
        "cluster_labels.npy",
    ]

    for fname in candidates:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            labels = np.load(path)
            print(f"Loaded labels from {path}: {labels.shape}")
            return labels, fname

    raise FileNotFoundError(
        f"None of the expected label files found in {data_dir}: "
        f"{', '.join(candidates)}"
    )

def compute_semantic_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    algorithm: str,
    save_path: str,
) -> Optional[float]:
    """
    Compute silhouette score using FINAL semantic labels.

    - Excludes noise points (label == -1).
    - Requires at least 2 clusters and at least 2 samples per cluster.
    - Saves a short text report to save_path.
    """
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Embeddings (N={embeddings.shape[0]}) and labels (N={labels.shape[0]}) "
            "must have the same length."
        )

    print("\n" + "=" * 80)
    print("Computing silhouette score for FINAL semantic clusters (Option A)")
    print("=" * 80)

    n_total = labels.shape[0]
    noise_mask = labels == -1
    n_noise = int(np.sum(noise_mask))

    # Exclude noise points
    eval_mask = ~noise_mask
    n_eval = int(np.sum(eval_mask))

    if n_eval == 0:
        print("No non-noise points available after excluding label -1. Skipping.")
        return None

    emb_eval = embeddings[eval_mask]
    labels_eval = labels[eval_mask]

    unique_labels, counts = np.unique(labels_eval, return_counts=True)

    print(f"Total samples            : {n_total}")
    print(f"Noise samples (label -1) : {n_noise}")
    print(f"Samples used for eval    : {n_eval}")
    print(f"Unique semantic clusters : {len(unique_labels)}")

    # Need at least 2 clusters
    if len(unique_labels) < 2:
        print("Silhouette score skipped: need at least 2 clusters.")
        return None

    # Each cluster must have at least 2 samples
    if np.any(counts < 2):
        print(
            "Silhouette score skipped: at least one cluster has fewer than 2 samples."
        )
        return None

    # Use all available points (no subsampling)
    print("Using all non-noise samples for silhouette computation.")

    try:
        score = silhouette_score(
            emb_eval,
            labels_eval,
            sample_size=None,  # no subsampling
            random_state=None,  # not used when sample_size=None
        )
    except Exception as e:
        print(f"Silhouette score computation failed: {e}")
        return None

    print(f"\nSilhouette score (final semantic clusters, {algorithm}) = {score:.6f}")

    # Save report
    with open(save_path, "w") as f:
        f.write("Semantic Clustering Metrics (Final Labels / Option A)\n")
        f.write("=====================================================\n")
        f.write(f"Algorithm (for logging)          : {algorithm}\n")
        f.write(f"Label file used                  : {os.path.basename(save_path)}\n")
        f.write(f"Num samples (total)              : {n_total}\n")
        f.write(f"Num noise samples (label == -1)  : {n_noise}\n")
        f.write(f"Num samples used for silhouette  : {n_eval}\n")
        f.write(f"Num semantic clusters (non-noise): {len(unique_labels)}\n")
        f.write(f"Silhouette score                 : {score:.6f}\n")

    print(f"\nSaved semantic clustering metrics to {save_path}")
    return score


def main():
    args = parse_args()

    data_dir = args.data_dir
    algorithm = args.algorithm

    print("=" * 80)
    print(f"Computing semantic silhouette in {data_dir}")
    print("=" * 80)

    embeddings = load_embeddings(data_dir)
    labels, labels_fname = load_final_labels(data_dir)

    metrics_path = os.path.join(data_dir, "semantic_cluster_metrics.txt")
    score = compute_semantic_silhouette(
        embeddings=embeddings,
        labels=labels,
        algorithm=algorithm,
        save_path=metrics_path,
    )

    if score is None:
        print("\nSilhouette score could not be computed (see messages above).")
    else:
        print(f"\nDone. Silhouette score: {score:.6f}")


if __name__ == "__main__":
    main()