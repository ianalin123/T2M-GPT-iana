"""
Test script to validate the cross-dataset classification workflow.

This script tests the workflow by:
1. Training a classifier on a subset of data
2. Saving the classifier
3. Loading it back and applying to held-out data
4. Verifying predictions work correctly

This simulates the HumanML → KIT workflow.
"""

import numpy as np
import os
import sys
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def test_workflow(data_dir="clustering/outputs"):
    """Test the complete workflow with existing data."""
    
    print("="*80)
    print("Testing Cross-Dataset Classification Workflow")
    print("="*80)
    
    # Check if test data exists
    embeddings_path = f"{data_dir}/embeddings_processed.npy"
    labels_path = f"{data_dir}/cluster_labels_denoised.npy"
    
    if not os.path.exists(embeddings_path):
        print(f"❌ Error: {embeddings_path} not found")
        print("Please run extract_embeddings.py first")
        return False
    
    if not os.path.exists(labels_path):
        print(f"❌ Error: {labels_path} not found")
        print("Please run clustering first")
        return False
    
    print("\n✓ Found required data files")
    
    # Step 1: Load data
    print("\n" + "="*80)
    print("Step 1: Loading Data")
    print("="*80)
    
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Aggregate if 3D
    if embeddings.ndim == 3:
        embeddings = embeddings.mean(axis=1)
        print(f"Aggregated to: {embeddings.shape}")
    
    # Split data to simulate two datasets (HumanML and KIT)
    X_humanml, X_kit, y_humanml, y_kit = train_test_split(
        embeddings, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"\nSimulated datasets:")
    print(f"  'HumanML' (training): {X_humanml.shape[0]} samples")
    print(f"  'KIT' (inference): {X_kit.shape[0]} samples")
    
    # Step 2: Train classifier on "HumanML"
    print("\n" + "="*80)
    print("Step 2: Training Classifier on 'HumanML'")
    print("="*80)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_humanml, y_humanml)
    
    train_score = clf.score(X_humanml, y_humanml)
    print(f"Training accuracy: {train_score:.3f}")
    
    # Step 3: Save classifier
    print("\n" + "="*80)
    print("Step 3: Saving Classifier")
    print("="*80)
    
    test_classifier_path = f"{data_dir}/test_classifier.joblib"
    dump(clf, test_classifier_path)
    print(f"✓ Saved to: {test_classifier_path}")
    
    # Step 4: Load classifier (simulating fresh session)
    print("\n" + "="*80)
    print("Step 4: Loading Saved Classifier")
    print("="*80)
    
    clf_loaded = load(test_classifier_path)
    print(f"✓ Loaded classifier: {type(clf_loaded).__name__}")
    print(f"  Expected features: {clf_loaded.n_features_in_}")
    print(f"  Expected classes: {clf_loaded.n_classes_}")
    
    # Step 5: Apply to "KIT"
    print("\n" + "="*80)
    print("Step 5: Applying to 'KIT' Dataset")
    print("="*80)
    
    # Check dimension compatibility
    if clf_loaded.n_features_in_ != X_kit.shape[1]:
        print(f"❌ Dimension mismatch!")
        return False
    
    print(f"✓ Dimensions match: {X_kit.shape[1]} features")
    
    # Predict
    kit_predictions = clf_loaded.predict(X_kit)
    print(f"Generated predictions for {len(kit_predictions)} samples")
    
    # Get prediction probabilities
    if hasattr(clf_loaded, 'predict_proba'):
        kit_probs = clf_loaded.predict_proba(X_kit)
        avg_confidence = kit_probs.max(axis=1).mean()
        print(f"Average prediction confidence: {avg_confidence:.3f}")
    
    # Step 6: Evaluate (in real scenario, KIT labels would be unknown)
    print("\n" + "="*80)
    print("Step 6: Evaluation (for testing purposes)")
    print("="*80)
    
    kit_accuracy = accuracy_score(y_kit, kit_predictions)
    print(f"'KIT' accuracy: {kit_accuracy:.3f}")
    print(f"(In real use, KIT labels would be unknown)")
    
    # Check prediction distribution
    unique_preds, pred_counts = np.unique(kit_predictions, return_counts=True)
    print(f"\nPrediction distribution:")
    print(f"  Unique predicted clusters: {len(unique_preds)}")
    print(f"  Coverage: {len(unique_preds)}/{clf_loaded.n_classes_} classes")
    
    # Step 7: Save test predictions
    print("\n" + "="*80)
    print("Step 7: Saving Results")
    print("="*80)
    
    test_predictions_path = f"{data_dir}/test_predictions.npy"
    np.save(test_predictions_path, kit_predictions)
    print(f"✓ Saved predictions to: {test_predictions_path}")
    
    # Cleanup test files
    print("\n" + "="*80)
    print("Cleanup")
    print("="*80)
    
    os.remove(test_classifier_path)
    os.remove(test_predictions_path)
    print("✓ Removed test files")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nThe workflow is ready to use:")
    print("1. train_supervised_clustering_example.py - trains and saves classifier")
    print("2. apply_classifier_to_kit.py - applies to new dataset")
    print("\nSee CROSS_DATASET_CLASSIFICATION.md for detailed instructions")
    
    return True


if __name__ == "__main__":
    success = test_workflow()
    sys.exit(0 if success else 1)

