"""
Script to apply a HumanML-trained supervised classifier to KIT embeddings.

This script assumes:
1. You've already trained a classifier on HumanML data using train_supervised_clustering_example.py
2. You've extracted KIT embeddings using the same encoder and layer as HumanML
3. The KIT embeddings have the same dimensionality as HumanML embeddings

Usage:
    python clustering/apply_classifier_to_kit.py \
        --classifier-path clustering/outputs/humanml_supervised_classifier_random_forest.joblib \
        --kit-embeddings-path clustering/outputs_kit/embeddings_processed.npy \
        --output-path clustering/outputs_kit/predicted_humanml_clusters.npy
"""

import numpy as np
import argparse
import os
from joblib import load
from collections import Counter


def load_embeddings_only(path: str):
    """
    Load embeddings and aggregate over time dimension if needed.
    
    This uses the same preprocessing as train_supervised_clustering_example.py
    to ensure consistency between training (HumanML) and inference (KIT).
    """
    print(f"\nLoading embeddings from: {path}")
    emb = np.load(path)
    print(f"Loaded embeddings shape: {emb.shape}")

    # Match the HumanML pipeline: mean over time if 3D (N, T, D)
    if emb.ndim == 3:
        emb = emb.mean(axis=1)
        print(f"Mean-pooled over time dimension: {emb.shape}")
    elif emb.ndim == 2:
        print(f"Using embeddings as-is (already 2D).")
    else:
        raise ValueError(f"Unexpected embedding dimensionality: {emb.ndim}D. Expected 2D or 3D.")
    
    return emb


def apply_classifier(clf, X_kit):
    """
    Apply trained classifier to KIT embeddings.
    
    Args:
        clf: Trained sklearn classifier
        X_kit: KIT embeddings (N, D) where D matches training data dimension
    
    Returns:
        predictions: Predicted cluster/verb labels for KIT samples
    """
    print(f"\nApplying classifier to {len(X_kit)} KIT samples...")
    predictions = clf.predict(X_kit)
    print(f"Predictions shape: {predictions.shape}")
    
    # Analyze prediction distribution
    pred_counts = Counter(predictions)
    n_clusters = len(pred_counts)
    
    print(f"\nPrediction Summary:")
    print(f"  Unique predicted clusters: {n_clusters}")
    print(f"  Total samples: {len(predictions)}")
    print(f"  Avg samples per cluster: {len(predictions) / n_clusters:.1f}")
    
    print(f"\nTop 10 most frequent predicted clusters:")
    for label, count in pred_counts.most_common(10):
        print(f"  Cluster {label:2d}: {count:5d} samples ({count/len(predictions)*100:5.1f}%)")
    
    return predictions


def get_prediction_probabilities(clf, X_kit):
    """
    Get prediction probabilities if the classifier supports it.
    
    Returns:
        probabilities: (N, n_classes) array of class probabilities, or None
    """
    if hasattr(clf, 'predict_proba'):
        print("\nComputing prediction probabilities...")
        probs = clf.predict_proba(X_kit)
        print(f"Probabilities shape: {probs.shape}")
        
        # Compute prediction confidence (max probability)
        max_probs = probs.max(axis=1)
        print(f"\nPrediction Confidence Statistics:")
        print(f"  Mean confidence: {max_probs.mean():.3f}")
        print(f"  Median confidence: {np.median(max_probs):.3f}")
        print(f"  Min confidence: {max_probs.min():.3f}")
        print(f"  Max confidence: {max_probs.max():.3f}")
        
        return probs
    else:
        print("\nClassifier does not support probability predictions (e.g., SVM without probability=True)")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Apply HumanML-trained classifier to KIT embeddings'
    )
    parser.add_argument(
        '--classifier-path',
        required=True,
        type=str,
        help='Path to saved classifier (.joblib file from train_supervised_clustering_example.py)',
    )
    parser.add_argument(
        '--kit-embeddings-path',
        required=True,
        type=str,
        help='Path to KIT embeddings file (.npy)',
    )
    parser.add_argument(
        '--output-path',
        default=None,
        type=str,
        help='Path to save predicted labels (.npy). If not provided, defaults to same dir as embeddings.',
    )
    parser.add_argument(
        '--save-probabilities',
        action='store_true',
        help='Also save prediction probabilities if classifier supports it',
    )
    parser.add_argument(
        '--texts-file',
        default=None,
        type=str,
        help='Optional: Path to text descriptions file to visualize predictions',
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.classifier_path):
        raise FileNotFoundError(f"Classifier not found: {args.classifier_path}")
    if not os.path.exists(args.kit_embeddings_path):
        raise FileNotFoundError(f"KIT embeddings not found: {args.kit_embeddings_path}")
    
    # Set default output path if not provided
    if args.output_path is None:
        kit_dir = os.path.dirname(args.kit_embeddings_path)
        args.output_path = os.path.join(kit_dir, 'predicted_humanml_clusters.npy')
    
    print("="*80)
    print("Applying HumanML Classifier to KIT Embeddings")
    print("="*80)
    print(f"\nInput Configuration:")
    print(f"  Classifier: {args.classifier_path}")
    print(f"  KIT embeddings: {args.kit_embeddings_path}")
    print(f"  Output: {args.output_path}")
    
    # 1. Load trained classifier
    print("\n" + "="*80)
    print("Step 1: Loading Trained Classifier")
    print("="*80)
    clf = load(args.classifier_path)
    print(f"✓ Loaded classifier: {type(clf).__name__}")
    
    # 2. Load KIT embeddings
    print("\n" + "="*80)
    print("Step 2: Loading KIT Embeddings")
    print("="*80)
    X_kit = load_embeddings_only(args.kit_embeddings_path)
    
    # Validate dimensionality
    expected_dim = clf.n_features_in_
    actual_dim = X_kit.shape[1]
    print(f"\nDimensionality check:")
    print(f"  Classifier expects: {expected_dim} features")
    print(f"  KIT embeddings have: {actual_dim} features")
    
    if expected_dim != actual_dim:
        raise ValueError(
            f"Dimension mismatch! Classifier trained on {expected_dim}-dim features, "
            f"but KIT embeddings have {actual_dim} dimensions. "
            f"Ensure KIT embeddings were extracted with the same encoder and layer as HumanML."
        )
    print("✓ Dimensions match!")
    
    # 3. Apply classifier
    print("\n" + "="*80)
    print("Step 3: Generating Predictions")
    print("="*80)
    kit_predictions = apply_classifier(clf, X_kit)
    
    # 4. Get probabilities if requested and supported
    probs = None
    if args.save_probabilities:
        probs = get_prediction_probabilities(clf, X_kit)
    
    # 5. Save predictions
    print("\n" + "="*80)
    print("Step 4: Saving Results")
    print("="*80)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, kit_predictions)
    print(f"✓ Saved predictions to: {args.output_path}")
    
    if probs is not None:
        probs_path = args.output_path.replace('.npy', '_probabilities.npy')
        np.save(probs_path, probs)
        print(f"✓ Saved probabilities to: {probs_path}")
    
    # 6. Optional: Visualize some predictions with text descriptions
    if args.texts_file and os.path.exists(args.texts_file):
        print("\n" + "="*80)
        print("Step 5: Sample Predictions")
        print("="*80)
        
        with open(args.texts_file, 'r') as f:
            texts = f.read().strip().split('\n')
        
        print(f"\nShowing 10 random predictions with text descriptions:\n")
        indices = np.random.choice(len(kit_predictions), min(10, len(kit_predictions)), replace=False)
        
        for i, idx in enumerate(indices, 1):
            text = texts[idx] if idx < len(texts) else "N/A"
            pred_label = kit_predictions[idx]
            
            if probs is not None:
                confidence = probs[idx].max()
                print(f"{i:2d}. [{pred_label:2d}] (conf: {confidence:.2f}) | {text[:80]}")
            else:
                print(f"{i:2d}. [{pred_label:2d}] | {text[:80]}")
    
    # Final summary
    print("\n" + "="*80)
    print("✓ Successfully Applied Classifier to KIT Dataset!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"  1. Analyze the predicted clusters: {args.output_path}")
    print(f"  2. You can now use these labels for:")
    print(f"     - Analyzing KIT dataset structure")
    print(f"     - Comparing KIT and HumanML motion distributions")
    print(f"     - Filtering/grouping KIT samples by predicted motion type")
    print(f"     - Transfer learning experiments")


if __name__ == '__main__':
    main()

