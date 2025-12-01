import argparse
import os
from collections import Counter, defaultdict

import numpy as np


def load_cluster_labels_and_verbs(data_dir: str):
    """
    Load original cluster labels and the cluster_id → atomic verb mapping.
    """
    labels_path = os.path.join(data_dir, "cluster_labels.npy")
    verb_labels_path = os.path.join(data_dir, "cluster_verb_labels.txt")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"cluster_labels.npy not found in {data_dir}")
    if not os.path.exists(verb_labels_path):
        raise FileNotFoundError(f"cluster_verb_labels.txt not found in {data_dir}")

    labels = np.load(labels_path)

    cluster_to_verb = {}
    with open(verb_labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            cid_str, verb = parts
            try:
                cid = int(cid_str)
            except ValueError:
                continue
            cluster_to_verb[cid] = verb

    return labels, cluster_to_verb


def compute_cluster_sizes(labels: np.ndarray):
    """
    Return a dict cluster_id → size (excluding noise label -1).
    """
    unique, counts = np.unique(labels, return_counts=True)
    sizes = {}
    for cid, cnt in zip(unique, counts):
        sizes[int(cid)] = int(cnt)
    return sizes


def merge_clusters_by_verb(
    labels: np.ndarray,
    cluster_to_verb: dict,
    target_verbs=None,
):
    """
    Merge clusters that share the same majority verb into a canonical cluster
    per verb (largest cluster becomes canonical).

    Args:
        labels: (N,) original cluster assignments (may include -1 for noise)
        cluster_to_verb: dict[int, str], cluster_id → verb
        target_verbs: optional iterable of verbs to merge; if None, all verbs

    Returns:
        labels_merged: new labels array with merged cluster IDs
        merge_report: dict with information about merges performed
    """
    if target_verbs is not None:
        target_verbs = set(target_verbs)

    sizes = compute_cluster_sizes(labels)

    # verb → list of cluster_ids
    verb_to_clusters = defaultdict(list)
    for cid, verb in cluster_to_verb.items():
        if cid == -1:
            continue
        if target_verbs is not None and verb not in target_verbs:
            continue
        verb_to_clusters[verb].append(cid)

    labels_merged = labels.copy()
    merge_report = {}

    for verb, cids in verb_to_clusters.items():
        if len(cids) <= 1:
            continue

        # pick canonical cluster: largest size
        canonical = max(cids, key=lambda c: sizes.get(c, 0))
        to_merge = [c for c in cids if c != canonical]

        if not to_merge:
            continue

        before_counts = {cid: sizes.get(cid, 0) for cid in cids}

        for cid in to_merge:
            mask = labels_merged == cid
            labels_merged[mask] = canonical

        after_sizes = compute_cluster_sizes(labels_merged)
        canonical_size_after = after_sizes.get(canonical, 0)

        merge_report[verb] = {
            "canonical": canonical,
            "merged_from": to_merge,
            "before_sizes": before_counts,
            "after_size": canonical_size_after,
        }

    return labels_merged, merge_report


def load_embeddings(data_dir: str):
    """
    Load embeddings_processed.npy for centroid-based operations.
    """
    emb_path = os.path.join(data_dir, "embeddings_processed.npy")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"embeddings_processed.npy not found in {data_dir}")
    return np.load(emb_path)


def compute_cluster_centroids(embeddings: np.ndarray, labels: np.ndarray, valid_clusters):
    """
    Compute centroids and intra-cluster distance thresholds for given clusters.

    Returns:
        centroids: dict[cid] = centroid vector
        distance_thresholds: dict[cid] = distance threshold (percentile-based)
    """
    centroids = {}
    distance_thresholds = {}

    for cid in valid_clusters:
        mask = labels == cid
        if not np.any(mask):
            continue
        points = embeddings[mask]
        centroid = points.mean(axis=0)
        centroids[cid] = centroid

        # distances of in-cluster points to centroid
        dists = np.linalg.norm(points - centroid, axis=1)
        threshold = np.percentile(dists, 90.0)  # default 90th percentile
        distance_thresholds[cid] = float(threshold)

    return centroids, distance_thresholds


def reassign_noise_to_nearest_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_to_verb: dict,
    target_verbs=None,
):
    """
    Reassign a subset of noise points (label -1) to nearest semantic clusters
    if they lie within a safe distance threshold from the cluster centroid.

    Args:
        embeddings: (N, D) processed embeddings
        labels: (N,) labels after merging
        cluster_to_verb: dict[int, str], original cluster_id → verb
        target_verbs: optional list of verbs to which noise can be reassigned;
                      if None, allow all non-noise clusters

    Returns:
        labels_denoised: new labels array with some noise reassigned
        stats: dict with reassignment statistics
    """
    n_samples = labels.shape[0]
    labels_denoised = labels.copy()

    if target_verbs is not None:
        target_verbs = set(target_verbs)

    # Determine which clusters are candidates for reassignment
    candidate_clusters = []
    for cid in np.unique(labels):
        if cid == -1:
            continue
        verb = cluster_to_verb.get(int(cid), None)
        if target_verbs is not None and verb not in target_verbs:
            continue
        candidate_clusters.append(int(cid))

    if not candidate_clusters:
        return labels_denoised, {
            "n_noise_before": int(np.sum(labels == -1)),
            "n_noise_after": int(np.sum(labels_denoised == -1)),
            "n_reassigned": 0,
        }

    centroids, distance_thresholds = compute_cluster_centroids(
        embeddings, labels_denoised, candidate_clusters
    )

    noise_indices = np.where(labels_denoised == -1)[0]
    n_noise_before = int(noise_indices.size)
    n_reassigned = 0

    for idx in noise_indices:
        point = embeddings[idx]

        best_cid = None
        best_dist = None

        for cid, centroid in centroids.items():
            dist = float(np.linalg.norm(point - centroid))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_cid = cid

        if best_cid is None:
            continue

        threshold = distance_thresholds.get(best_cid, None)
        if threshold is None:
            continue

        if best_dist <= threshold:
            labels_denoised[idx] = best_cid
            n_reassigned += 1

    n_noise_after = int(np.sum(labels_denoised == -1))

    stats = {
        "n_samples": n_samples,
        "n_noise_before": n_noise_before,
        "n_noise_after": n_noise_after,
        "n_reassigned": n_reassigned,
    }

    return labels_denoised, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Post-process clustering results: merge repeated verb clusters "
            "into a single cluster per verb and optionally reassign some noise."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="clustering/outputs",
        help="Directory containing cluster_labels.npy, cluster_verb_labels.txt, "
        "and embeddings_processed.npy.",
    )
    parser.add_argument(
        "--verbs",
        type=str,
        nargs="*",
        default=["walk", "raise", "pick"],
        help="Verbs for which to merge repeated clusters and (optionally) "
        "reassign noise. Default: walk raise pick.",
    )
    parser.add_argument(
        "--no-reassign-noise",
        action="store_true",
        help="If set, only merge clusters and do not reassign noise.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print(f"Loading clustering outputs from {args.data_dir} ...")
    print("=" * 80)

    labels, cluster_to_verb = load_cluster_labels_and_verbs(args.data_dir)
    print(f"Loaded labels shape: {labels.shape}")

    unique, counts = np.unique(labels, return_counts=True)
    print("Original label distribution:")
    for cid, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        if cid == -1:
            print(f"  Noise   : {cnt:6d} samples")
        else:
            verb = cluster_to_verb.get(int(cid), "unknown")
            print(f"  Cluster {int(cid):2d} ({verb:8s}): {cnt:6d} samples")

    # Step 1: merge clusters for specific verbs
    labels_merged, merge_report = merge_clusters_by_verb(
        labels, cluster_to_verb, target_verbs=args.verbs
    )

    merged_path = os.path.join(args.data_dir, "cluster_labels_merged.npy")
    np.save(merged_path, labels_merged)
    print(f"\nSaved merged labels to {merged_path}")

    if merge_report:
        print("\nMerge summary (by verb):")
        for verb, info in merge_report.items():
            canonical = info["canonical"]
            merged_from = info["merged_from"]
            before_sizes = info["before_sizes"]
            after_size = info["after_size"]

            print(f"  Verb '{verb}':")
            print(f"    Canonical cluster: {canonical}")
            print(f"    Merged from      : {merged_from}")
            print(f"    Sizes before     : {before_sizes}")
            print(f"    Canonical size after merge: {after_size}")
    else:
        print("\nNo clusters were merged for the specified verbs.")

    # Step 2: optionally reassign noise
    if args.no_reassign_noise:
        return

    print("\n" + "=" * 80)
    print("Reassigning a subset of noise points to nearest semantic clusters ...")
    print("=" * 80)

    embeddings = load_embeddings(args.data_dir)
    if embeddings.shape[0] != labels_merged.shape[0]:
        raise ValueError(
            f"Mismatch between embeddings ({embeddings.shape[0]}) and labels "
            f"({labels_merged.shape[0]})"
        )

    labels_denoised, noise_stats = reassign_noise_to_nearest_clusters(
        embeddings,
        labels_merged,
        cluster_to_verb,
        target_verbs=args.verbs,
    )

    denoised_path = os.path.join(args.data_dir, "cluster_labels_denoised.npy")
    np.save(denoised_path, labels_denoised)
    print(f"\nSaved denoised labels to {denoised_path}")

    print(
        "\nNoise reassignment statistics:\n"
        f"  Total samples          : {noise_stats['n_samples']}\n"
        f"  Noise before           : {noise_stats['n_noise_before']} "
        f"({noise_stats['n_noise_before']/max(1, noise_stats['n_samples'])*100:.2f}%)\n"
        f"  Noise after            : {noise_stats['n_noise_after']} "
        f"({noise_stats['n_noise_after']/max(1, noise_stats['n_samples'])*100:.2f}%)\n"
        f"  Reassigned from noise  : {noise_stats['n_reassigned']}"
    )


if __name__ == "__main__":
    main()


