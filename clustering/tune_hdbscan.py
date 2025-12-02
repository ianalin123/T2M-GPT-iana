import argparse
import os
from collections import Counter, defaultdict

import numpy as np


def load_embeddings_and_labels(data_dir: str):
    """
    Load processed embeddings, original cluster labels and cluster→verb mapping.

    We use:
      - embeddings_processed.npy: reduced embeddings used for clustering/visualization
      - cluster_labels.npy: original clustering labels (e.g. HDBSCAN or K-Means)
      - cluster_verb_labels.txt: mapping cluster_id → majority atomic verb

    These let us approximate a per-sample atomic verb:
      sample_verb[i] = cluster_verb_labels[ cluster_labels[i] ]
    which we then use to evaluate verb purity for new HDBSCAN runs.
    """
    emb_path = os.path.join(data_dir, "embeddings_processed.npy")
    labels_path = os.path.join(data_dir, "cluster_labels.npy")
    verb_labels_path = os.path.join(data_dir, "cluster_verb_labels.txt")

    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"embeddings_processed.npy not found in {data_dir}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"cluster_labels.npy not found in {data_dir}")
    if not os.path.exists(verb_labels_path):
        raise FileNotFoundError(f"cluster_verb_labels.txt not found in {data_dir}")

    embeddings = np.load(emb_path)
    orig_labels = np.load(labels_path)

    if embeddings.shape[0] != orig_labels.shape[0]:
        raise ValueError(
            f"Mismatch between embeddings ({embeddings.shape[0]}) "
            f"and labels ({orig_labels.shape[0]})"
        )

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

    # Map each sample to its majority atomic verb from the original clustering
    sample_verbs = []
    for c in orig_labels:
        if c == -1:
            sample_verbs.append("noise")
        else:
            sample_verbs.append(cluster_to_verb.get(int(c), "unknown"))

    return embeddings, np.array(sample_verbs), orig_labels, cluster_to_verb


def evaluate_clustering(labels: np.ndarray, sample_verbs: np.ndarray):
    """
    Compute summary metrics for a clustering:
      - fraction of noise points
      - number of clusters (excluding noise)
      - average verb purity across clusters (size-weighted)
      - per-cluster majority verb and purity
    """
    assert labels.shape[0] == sample_verbs.shape[0]

    n_samples = labels.shape[0]
    noise_mask = labels == -1
    n_noise = int(noise_mask.sum())
    frac_noise = n_noise / float(n_samples)

    unique_clusters = sorted(c for c in np.unique(labels) if c != -1)
    n_clusters = len(unique_clusters)

    cluster_major_verb = {}
    cluster_purity = {}

    total_major_count = 0
    total_cluster_points = 0

    for cid in unique_clusters:
        mask = labels == cid
        verbs = sample_verbs[mask]
        if verbs.size == 0:
            continue
        counts = Counter(verbs)
        major_verb, major_count = counts.most_common(1)[0]
        purity = major_count / float(verbs.size)

        cluster_major_verb[cid] = major_verb
        cluster_purity[cid] = purity

        total_major_count += major_count
        total_cluster_points += verbs.size

    avg_verb_purity = 0.0
    if total_cluster_points > 0:
        avg_verb_purity = total_major_count / float(total_cluster_points)

    return {
        "n_samples": n_samples,
        "n_noise": n_noise,
        "frac_noise": frac_noise,
        "n_clusters": n_clusters,
        "avg_verb_purity": avg_verb_purity,
        "cluster_major_verb": cluster_major_verb,
        "cluster_purity": cluster_purity,
    }


def run_hdbscan_grid(
    embeddings: np.ndarray,
    sample_verbs: np.ndarray,
    min_cluster_sizes,
    min_samples_list,
    epsilons,
    cluster_selection_method="eom",
):
    """
    Run a small HDBSCAN hyperparameter sweep and print a concise summary table.
    """
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError(
            "hdbscan is not installed. Please install it in your environment, "
            "e.g. `pip install hdbscan` or via your conda env."
        ) from exc

    results = []

    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_list:
            for eps in epsilons:
                print(
                    f"\n=== HDBSCAN: min_cluster_size={min_cluster_size}, "
                    f"min_samples={min_samples}, epsilon={eps} ==="
                )

                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric="euclidean",
                    cluster_selection_method=cluster_selection_method,
                    cluster_selection_epsilon=eps,
                    prediction_data=False,
                )

                labels = clusterer.fit_predict(embeddings)

                metrics = evaluate_clustering(labels, sample_verbs)

                print(
                    f"  #clusters (excl. noise): {metrics['n_clusters']}\n"
                    f"  noise fraction         : {metrics['frac_noise']*100:.2f}% "
                    f"({metrics['n_noise']}/{metrics['n_samples']})\n"
                    f"  avg verb purity        : {metrics['avg_verb_purity']*100:.2f}%"
                )

                # Optional: quick breakdown for key verbs of interest
                verb_to_clusters = defaultdict(list)
                for cid, verb in metrics["cluster_major_verb"].items():
                    verb_to_clusters[verb].append(
                        (cid, metrics["cluster_purity"].get(cid, 0.0))
                    )

                for verb in ["walk", "raise", "pick"]:
                    if verb in verb_to_clusters:
                        clusters_info = ", ".join(
                            f"{cid} (purity={purity*100:.1f}%)"
                            for cid, purity in sorted(
                                verb_to_clusters[verb], key=lambda x: -x[1]
                            )
                        )
                        print(f"  {verb:5s} clusters       : {clusters_info}")

                results.append(
                    {
                        "min_cluster_size": min_cluster_size,
                        "min_samples": min_samples,
                        "epsilon": eps,
                        **metrics,
                    }
                )

    # Print a compact summary table sorted by low noise and high purity
    print("\n" + "=" * 80)
    print("GRID SEARCH SUMMARY (sorted by noise fraction ↑, avg verb purity ↓)")
    print("=" * 80)
    header = (
        "min_cs  min_s  eps    #clust  noise%   avg_purity%  n_noise/n   "
        "notes"
    )
    print(header)
    print("-" * len(header))

    def sort_key(r):
        # sort by noise fraction ascending, then by avg verb purity descending
        return (r["frac_noise"], -r["avg_verb_purity"])

    for r in sorted(results, key=sort_key):
        line = (
            f"{r['min_cluster_size']:6d}  "
            f"{str(r['min_samples']):5s}  "
            f"{r['epsilon']:<5.2f}  "
            f"{r['n_clusters']:6d}  "
            f"{r['frac_noise']*100:6.2f}  "
            f"{r['avg_verb_purity']*100:11.2f}  "
            f"{r['n_noise']:5d}/{r['n_samples']:<5d}  "
            f""
        )
        print(line)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Tune HDBSCAN hyperparameters on precomputed embeddings to "
            "reduce noise and maintain high verb purity."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="clustering/outputs",
        help="Directory containing embeddings_processed.npy, "
        "cluster_labels.npy, cluster_verb_labels.txt",
    )
    parser.add_argument(
        "--min-cluster-sizes",
        type=int,
        nargs="+",
        default=[5, 7, 10, 12, 15],
        help="List of min_cluster_size values to try.",
    )
    parser.add_argument(
        "--min-samples-list",
        type=int,
        nargs="+",
        default=[None, 5, 10],
        help=(
            "List of min_samples values to try. Use None to indicate min_cluster_size"
        ),
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.2, 0.5, 0.8],
        help="List of cluster_selection_epsilon values to try.",
    )
    parser.add_argument(
        "--cluster-selection-method",
        type=str,
        default="eom",
        choices=["eom", "leaf", "dbscan"],
        help="HDBSCAN cluster_selection_method (default: eom).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Normalize min_samples_list: replace -1 with None
    min_samples_list = [
        None if (v is None or isinstance(v, int) and v < 0) else v
        for v in args.min_samples_list
    ]

    print("=" * 80)
    print(f"Loading embeddings and labels from {args.data_dir} ...")
    print("=" * 80)
    embeddings, sample_verbs, orig_labels, cluster_to_verb = load_embeddings_and_labels(
        args.data_dir
    )
    print(f"Embeddings shape      : {embeddings.shape}")
    print(f"Unique original labels: {len(np.unique(orig_labels))}")

    print("\nSample of original cluster→verb mapping:")
    for cid in sorted(cluster_to_verb.keys())[:10]:
        print(f"  Cluster {cid:2d}: {cluster_to_verb[cid]}")

    print("\n" + "=" * 80)
    print("Running HDBSCAN hyperparameter sweep ...")
    print("=" * 80)
    run_hdbscan_grid(
        embeddings,
        sample_verbs,
        min_cluster_sizes=args.min_cluster_sizes,
        min_samples_list=min_samples_list,
        epsilons=args.epsilons,
        cluster_selection_method=args.cluster_selection_method,
    )


if __name__ == "__main__":
    main()



