"""
Example script: Training supervised/semi-supervised clustering using verb labels

This demonstrates how to use the reduced verb labels (from any clustering method)
as supervision for training various clustering and classification algorithms.
"""

import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
from joblib import dump


def load_clustering_data(
    data_dir: str = "clustering/outputs",
    embeddings_file: str = "encoder_embeddings.npy",
    labels_file: str = "verb_cluster_labels.npy",
    kmeans_labels_file: str = "cluster_labels.npy",
):
    """
    Load embeddings and labels from clustering analysis.

    This function is flexible enough to support:
      - Legacy setup: encoder_embeddings.npy + verb_cluster_labels.npy
      - New setup: embeddings_processed.npy + cluster_labels_denoised.npy
    """

    print(f"Loading data from {data_dir}...")

    # Load embeddings
    emb_path = f"{data_dir}/{embeddings_file}"
    # Use allow_pickle=True to support embeddings saved with pickling
    embeddings = np.load(emb_path, allow_pickle=True)
    print(f"Loaded embeddings: {embeddings.shape} from {emb_path}")

    # Aggregate over time dimension (mean pooling) only if 3D (N, T, D)
    if embeddings.ndim == 3:
        embeddings_agg = embeddings.mean(axis=1)
        print(f"Aggregated embeddings (mean over time): {embeddings_agg.shape}")
    else:
        embeddings_agg = embeddings
        print(f"Using embeddings as-is (2D): {embeddings_agg.shape}")

    # Load primary labels (verb- or cluster-based)
    labels_path = f"{data_dir}/{labels_file}"
    # Use allow_pickle=True to support object arrays (e.g., merged/semantic label files)
    verb_labels = np.load(labels_path, allow_pickle=True)
    print(f"Loaded labels from {labels_path}: {verb_labels.shape}")
    print(f"Number of unique labels: {len(np.unique(verb_labels))}")

    # Load optional K-means labels (for comparison)
    kmeans_labels = None
    kmeans_path = f"{data_dir}/{kmeans_labels_file}" if kmeans_labels_file else None
    if kmeans_path and os.path.exists(kmeans_path):
        kmeans_labels = np.load(kmeans_path)

    # Load text descriptions (optional)
    texts_path = f"{data_dir}/texts.txt"
    texts = []
    if os.path.exists(texts_path):
        with open(texts_path, "r") as f:
            texts = f.read().strip().split("\n")

    return embeddings_agg, verb_labels, kmeans_labels, texts


def train_supervised_classifier(X, y, classifier_type='random_forest'):
    """
    Train a supervised classifier using verb labels
    
    This treats the verb clusters as ground truth labels and trains
    a classifier to predict them from motion embeddings.
    """
    print(f"\n{'='*80}")
    print(f"Training Supervised Classifier: {classifier_type}")
    print(f"{'='*80}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Choose classifier
    if classifier_type == 'logistic':
        clf = LogisticRegression(max_iter=10000, random_state=42, n_jobs=-1)
    elif classifier_type == 'svm':
        clf = SVC(kernel='linear', random_state=42, C=1.0)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")
    
    # Train
    print(f"\nTraining {classifier_type}...")
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\nResults:")
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy:  {test_acc:.3f}")
    
    # Cross-validation
    print(f"\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
    print(f"  CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Detailed classification report
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test))
    
    return clf, y_pred_test, y_test


def train_semisupervised_clustering(X, y, labeled_fraction=0.1):
    """
    Semi-supervised learning: use only a fraction of labels
    
    This is useful when you want to leverage verb labels but only have
    labels for a subset of your data.
    """
    print(f"\n{'='*80}")
    print(f"Semi-Supervised Learning (Label Propagation)")
    print(f"{'='*80}")
    print(f"Using {labeled_fraction*100:.0f}% of labels as supervision")
    
    # Create partially labeled dataset
    n_samples = len(y)
    n_labeled = int(n_samples * labeled_fraction)
    
    # Randomly select which samples to label
    labeled_indices = np.random.choice(n_samples, n_labeled, replace=False)
    
    # Create y_partial where -1 means unlabeled
    y_partial = np.full_like(y, -1)
    y_partial[labeled_indices] = y[labeled_indices]
    
    print(f"Labeled samples: {n_labeled}")
    print(f"Unlabeled samples: {n_samples - n_labeled}")
    
    # Label Propagation
    print("\nTraining Label Propagation...")
    label_prop = LabelPropagation(kernel='rbf', max_iter=1000)
    label_prop.fit(X, y_partial)
    
    y_pred = label_prop.predict(X)
    
    # Evaluate on the unlabeled portion
    unlabeled_mask = y_partial == -1
    unlabeled_acc = accuracy_score(y[unlabeled_mask], y_pred[unlabeled_mask])
    
    print(f"\nResults:")
    print(f"  Accuracy on unlabeled data: {unlabeled_acc:.3f}")
    
    # Compare with fully supervised
    labeled_mask = y_partial != -1
    labeled_acc = accuracy_score(y[labeled_mask], y_pred[labeled_mask])
    print(f"  Accuracy on labeled data: {labeled_acc:.3f}")
    
    return label_prop, y_pred


def compare_with_unsupervised(X, verb_labels, kmeans_labels):
    """
    Compare unsupervised K-means with verb-based clusters
    
    This helps understand how well unsupervised clustering aligns
    with semantic verb categories.
    """
    print(f"\n{'='*80}")
    print("Comparing K-means with Verb-based Clusters")
    print(f"{'='*80}")
    
    # Compute agreement metrics
    ari = adjusted_rand_score(verb_labels, kmeans_labels)
    nmi = normalized_mutual_info_score(verb_labels, kmeans_labels)
    
    print(f"\nAgreement between K-means and verb clusters:")
    print(f"  Adjusted Rand Index: {ari:.3f}")
    print(f"  Normalized Mutual Information: {nmi:.3f}")
    
    print(f"\nInterpretation:")
    if ari > 0.5:
        print(f"  High agreement - K-means discovers verb-like clusters naturally")
    elif ari > 0.2:
        print(f"  Moderate agreement - Some alignment with verb semantics")
    else:
        print(f"  Low agreement - K-means and verb clusters are quite different")
    
    return ari, nmi


def visualize_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Visualize confusion matrix"""
    
    print(f"\nGenerating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix: Predicted vs True Verb Clusters')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {save_path}")
    plt.close()


def analyze_label_distribution(verb_labels):
    """Analyze and visualize label distribution"""
    
    print(f"\n{'='*80}")
    print("Label Distribution Analysis")
    print(f"{'='*80}")
    
    label_counts = Counter(verb_labels)
    n_clusters = len(label_counts)
    
    print(f"\nTotal clusters: {n_clusters}")
    print(f"Total samples: {len(verb_labels)}")
    print(f"Avg samples per cluster: {len(verb_labels) / n_clusters:.1f}")
    
    print(f"\nCluster sizes (top 10):")
    for label, count in label_counts.most_common(10):
        print(f"  Cluster {label:2d}: {count:5d} samples ({count/len(verb_labels)*100:5.1f}%)")
    
    # Check for imbalanced classes
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\nClass imbalance:")
    print(f"  Largest cluster: {max_count} samples")
    print(f"  Smallest cluster: {min_count} samples")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}x")
    
    if imbalance_ratio > 10:
        print(f"  ⚠️ Warning: Significant class imbalance detected!")
        print(f"     Consider using stratified sampling or class weights")
    
    # Visualize distribution
    plt.figure(figsize=(14, 6))
    counts = [label_counts[i] for i in sorted(label_counts.keys())]
    plt.bar(range(len(counts)), counts)
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Samples')
    plt.title('Verb Cluster Distribution')
    plt.tight_layout()
    plt.savefig('cluster_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved distribution plot to cluster_distribution.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Train supervised/semi-supervised models using verb cluster labels'
    )
    parser.add_argument('--data-dir', default='clustering/outputs', type=str,
                       help='Directory containing clustering results')
    parser.add_argument(
        '--embeddings-file',
        default='embeddings_processed.npy',
        type=str,
        help='NumPy file with embeddings (e.g., embeddings_processed.npy or encoder_embeddings.npy)',
    )
    parser.add_argument(
        '--labels-file',
        default='cluster_labels_denoised.npy',
        type=str,
        help='NumPy file with labels (e.g., cluster_labels_denoised.npy or verb_cluster_labels.npy)',
    )
    parser.add_argument(
        '--kmeans-labels-file',
        default='cluster_labels.npy',
        type=str,
        help='Optional NumPy file with K-Means labels for comparison (cluster_labels.npy)',
    )
    parser.add_argument('--classifier', default='random_forest',
                       choices=['random_forest', 'gradient_boosting', 'logistic', 'svm'],
                       help='Classifier type for supervised learning')
    parser.add_argument('--semisupervised', action='store_true',
                       help='Run semi-supervised experiment')
    parser.add_argument('--labeled-fraction', default=0.1, type=float,
                       help='Fraction of labels to use in semi-supervised learning')
    parser.add_argument('--compare-kmeans', action='store_true',
                       help='Compare with unsupervised K-means clustering')
    
    args = parser.parse_args()
    
    # Load data
    embeddings, verb_labels, kmeans_labels, texts = load_clustering_data(
        args.data_dir,
        embeddings_file=args.embeddings_file,
        labels_file=args.labels_file,
        kmeans_labels_file=args.kmeans_labels_file,
    )
    
    # Analyze label distribution
    analyze_label_distribution(verb_labels)
    
    # 1. Supervised classification
    print(f"\n{'='*80}")
    print("EXPERIMENT 1: Supervised Classification")
    print(f"{'='*80}")
    clf, y_pred, y_test = train_supervised_classifier(
        embeddings, verb_labels, 
        classifier_type=args.classifier
    )
    
    # Save trained classifier
    classifier_save_path = f'{args.data_dir}/humanml_supervised_classifier_{args.classifier}.joblib'
    dump(clf, classifier_save_path)
    print(f"\n✓ Saved trained classifier to {classifier_save_path}")
    
    # Visualize confusion matrix
    visualize_confusion_matrix(y_test, y_pred, 
                               save_path=f'confusion_matrix_{args.classifier}.png')
    
    # 2. Semi-supervised learning
    if args.semisupervised:
        print(f"\n{'='*80}")
        print("EXPERIMENT 2: Semi-Supervised Learning")
        print(f"{'='*80}")
        label_prop, y_pred_semi = train_semisupervised_clustering(
            embeddings, verb_labels,
            labeled_fraction=args.labeled_fraction
        )
    
    # 3. Compare with K-means
    if args.compare_kmeans and kmeans_labels is not None:
        print(f"\n{'='*80}")
        print("EXPERIMENT 3: Comparison with Unsupervised K-means")
        print(f"{'='*80}")
        ari, nmi = compare_with_unsupervised(embeddings, verb_labels, kmeans_labels)
    
    print(f"\n{'='*80}")
    print("✓ All experiments completed!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

