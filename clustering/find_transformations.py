import h5py
import numpy as np
from itertools import permutations
from collections import defaultdict
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from pathlib import Path


def load_cluster_embeddings(filepath):
    """Load all embeddings organized by cluster."""
    cluster_embeddings = defaultdict(list)
    cluster_samples = defaultdict(list)

    with h5py.File(filepath, "r") as f:
        for file_id in f.keys():
            cluster_id = f[file_id].attrs["cluster_id"]
            cluster_label = f[file_id].attrs["cluster_label"]
            encoder_embeddings = f[file_id]["encoder_embeddings"][:]

            # Use mean pooling to get a single embedding per sample
            mean_embedding = np.mean(encoder_embeddings, axis=0)

            cluster_embeddings[cluster_label].append(mean_embedding)
            cluster_samples[cluster_label].append(file_id)

    return cluster_embeddings, cluster_samples


def get_cluster_centroid(cluster_embeddings, cluster_label):
    """Get the mean embedding (centroid) for a cluster."""
    embeddings = np.array(cluster_embeddings[cluster_label])
    return np.mean(embeddings, axis=0)


def compute_analogy(cluster_embeddings, A, B, C):
    """
    Compute analogy: A is to B as C is to ?
    Example: "walk" is to "run" as "raise" is to ?

    Returns the direction vector: B - A + C
    """
    centroid_A = get_cluster_centroid(cluster_embeddings, A)
    centroid_B = get_cluster_centroid(cluster_embeddings, B)
    centroid_C = get_cluster_centroid(cluster_embeddings, C)

    # Compute the analogy vector
    direction = centroid_B - centroid_A
    result_vector = centroid_C + direction

    return result_vector, direction


def find_nearest_cluster(
    result_vector, cluster_embeddings, exclude=None, metric="cosine"
):
    """Find the cluster whose centroid is closest to the result vector.

    Args:
        result_vector: Target vector
        cluster_embeddings: Dictionary of cluster embeddings
        exclude: List of cluster labels to exclude
        metric: 'cosine' for cosine similarity, 'correlation' for Pearson correlation,
                'euclidean' for Euclidean distance
    """
    exclude = exclude or []
    best_score = float("-inf") if metric in ["cosine", "correlation"] else float("inf")
    nearest_cluster = None

    for cluster_label in cluster_embeddings.keys():
        if cluster_label in exclude:
            continue

        centroid = get_cluster_centroid(cluster_embeddings, cluster_label)

        if metric == "cosine":
            score = cosine_similarity(result_vector, centroid)
        elif metric == "correlation":
            score = correlation(result_vector, centroid)
        else:  # euclidean
            score = np.linalg.norm(result_vector - centroid)

        if metric in ["cosine", "correlation"]:
            if score > best_score:
                best_score = score
                nearest_cluster = cluster_label
        else:
            if score < best_score:
                best_score = score
                nearest_cluster = cluster_label

    return nearest_cluster, best_score


def compute_cluster_direction(cluster_embeddings, cluster_A, cluster_B):
    """
    Compute the semantic direction from cluster A to cluster B.
    This gives you a direction vector you can add/subtract from other clusters.
    """
    centroid_A = get_cluster_centroid(cluster_embeddings, cluster_A)
    centroid_B = get_cluster_centroid(cluster_embeddings, cluster_B)

    direction = centroid_B - centroid_A
    return direction


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return 1 - cosine(vec1, vec2)


def correlation(vec1, vec2):
    """Compute Pearson correlation between two vectors."""
    corr, _ = pearsonr(vec1, vec2)
    return corr


if __name__ == "__main__":
    filepath = "clustering/outputs/embeddings_clustered_denoised.h5"

    print("Loading embeddings...")
    cluster_embeddings, cluster_samples = load_cluster_embeddings(filepath)
    cluster_labels = list(cluster_embeddings.keys())

    print("Creating report...")

    with open("clustering/outputs/transformations_report.txt", "w") as f:
        print(f"Loaded {len(cluster_embeddings)} clusters", file=f)
        print(f"Available clusters: {cluster_labels}\n", file=f)

        # Compute direction and transform
        triple_verbs = list(permutations(cluster_labels, r=3))

        for i in range(len(triple_verbs)):
            A, B, C = triple_verbs[i]

            direction = compute_cluster_direction(cluster_embeddings, A, B)

            transformee_centroid = get_cluster_centroid(cluster_embeddings, C)
            new_vector = transformee_centroid + direction

            analogy_res, dist = find_nearest_cluster(
                new_vector, cluster_embeddings, exclude=[C]
            )

            if "noise" in [A, B, C, analogy_res] or "step" in [A, B, C, analogy_res]:
                continue

            print(f"'{C}' + ('{A}' -> '{B}') = '{analogy_res}'", file=f)

            Path(f"clustering/transformations").mkdir(parents=True, exist_ok=True)

            with open(f"clustering/transformations/{i}.npy", "wb") as g:
                np.save(g, new_vector)

            with open(f"clustering/transformations/logs.txt", "a") as g:
                g.write(f"{i}: {analogy_res}\n")


        # Cosine similarity relationships

        print(f"{'':8}", end="", file=f)
        for c in cluster_labels:
            print(f"{c:8}", end="", file=f)
        print("", file=f)

        for c1 in cluster_labels:
            print(f"{c1:8}", end="", file=f)
            centroid1 = get_cluster_centroid(cluster_embeddings, c1)
            for c2 in cluster_labels:
                centroid2 = get_cluster_centroid(cluster_embeddings, c2)
                sim = cosine_similarity(centroid1, centroid2)
                print(f"{sim:8.2f}", end="", file=f)
            print("", file=f)
    
    print("Report created in clustering/outputs/transformations_report.txt")
