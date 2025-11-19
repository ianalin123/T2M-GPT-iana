"""
Utility functions for reducing many verb labels to a fixed number of clusters
"""

import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import KMeans


def cluster_by_primary_verb(names, verb_labels):
    """
    Cluster by the primary (first) verb only
    
    Args:
        names: list of sample names
        verb_labels: dict mapping filename to verb label
    
    Returns:
        verb_cluster_labels: (N,) array of cluster IDs
        verb_label_to_id: dict mapping primary verb to cluster ID
    """
    if verb_labels is None:
        return None, None
    
    verb_cluster_labels = []
    verb_label_to_id = {}
    
    for name in names:
        base_name = name.split('_')[0] if '_' in name else name
        full_verb_label = verb_labels.get(base_name, 'unknown')
        
        # Take only the first verb
        primary_verb = full_verb_label.split('-')[0] if '-' in full_verb_label else full_verb_label
        
        if primary_verb not in verb_label_to_id:
            verb_label_to_id[primary_verb] = len(verb_label_to_id)
        
        verb_cluster_labels.append(verb_label_to_id[primary_verb])
    
    print(f"\nPrimary verb clustering: {len(verb_label_to_id)} unique verbs")
    
    return np.array(verb_cluster_labels), verb_label_to_id


def cluster_by_frequent_verbs(names, verb_labels, num_clusters=20):
    """
    Keep top num_clusters-1 most frequent verb labels, group rest as 'other'
    
    Args:
        names: list of sample names
        verb_labels: dict mapping filename to verb label
        num_clusters: desired number of clusters
    
    Returns:
        verb_cluster_labels: (N,) array of cluster IDs
        verb_label_to_id: dict mapping verb label to cluster ID
    """
    if verb_labels is None:
        return None, None
    
    # Count verb label frequencies
    verb_counts = Counter()
    for name in names:
        base_name = name.split('_')[0] if '_' in name else name
        verb_label = verb_labels.get(base_name, 'unknown')
        verb_counts[verb_label] += 1
    
    print(f"\nTotal unique verb combinations: {len(verb_counts)}")
    print(f"Reducing to {num_clusters} clusters...")
    
    # Get top K-1 most frequent (save 1 slot for "other")
    top_verbs = [verb for verb, count in verb_counts.most_common(num_clusters - 1)]
    
    # Create mapping
    verb_label_to_id = {verb: i for i, verb in enumerate(top_verbs)}
    verb_label_to_id['other'] = len(top_verbs)
    
    # Count how many samples fall into "other"
    other_count = 0
    
    # Assign cluster IDs
    verb_cluster_labels = []
    for name in names:
        base_name = name.split('_')[0] if '_' in name else name
        verb_label = verb_labels.get(base_name, 'unknown')
        
        if verb_label in verb_label_to_id:
            cluster_id = verb_label_to_id[verb_label]
        else:
            cluster_id = verb_label_to_id['other']
            other_count += 1
        
        verb_cluster_labels.append(cluster_id)
    
    print(f"Top {len(top_verbs)} verb labels cover {len(verb_cluster_labels) - other_count} samples")
    print(f"'other' category contains {other_count} samples from {len(verb_counts) - len(top_verbs)} rare verb combinations")
    
    return np.array(verb_cluster_labels), verb_label_to_id


def cluster_verbs_semantic(names, verb_labels, num_clusters=20, glove_path='./glove', 
                           vab_name='our_vab', embedding_dim=300):
    """
    Cluster verb labels using word embeddings and K-means
    
    This groups semantically similar verb combinations together
    (e.g., "walk-run" and "jog-walk" might be in the same cluster)
    
    Args:
        names: list of sample names
        verb_labels: dict mapping filename to verb label
        num_clusters: desired number of clusters
        glove_path: path to GloVe embeddings
        vab_name: vocabulary name
        embedding_dim: dimension of word embeddings
    
    Returns:
        verb_cluster_labels: (N,) array of cluster IDs
        verb_label_to_cluster: dict mapping verb label to cluster ID
    """
    if verb_labels is None:
        return None, None
    
    try:
        from utils.word_vectorizer import WordVectorizer
        w_vectorizer = WordVectorizer(glove_path, vab_name)
        print(f"Loaded word embeddings from {glove_path}")
    except Exception as e:
        print(f"Warning: Could not load word vectorizer: {e}")
        print("Falling back to frequency-based clustering...")
        return cluster_by_frequent_verbs(names, verb_labels, num_clusters)
    
    # Get unique verb labels and their embeddings
    unique_labels = list(set(verb_labels.values()))
    print(f"\nClustering {len(unique_labels)} unique verb combinations into {num_clusters} clusters...")
    
    label_embeddings = []
    valid_labels = []
    
    for label in unique_labels:
        # Split verbs and average their embeddings
        verbs = label.split('-')
        verb_vecs = []
        
        for verb in verbs:
            # Try to get embedding for this verb
            try:
                if hasattr(w_vectorizer, 'word2idx') and verb in w_vectorizer.word2idx:
                    vec = w_vectorizer.glove[w_vectorizer.word2idx[verb]]
                    verb_vecs.append(vec)
            except:
                pass
        
        if verb_vecs:
            # Average embedding of all verbs in the combination
            avg_embedding = np.mean(verb_vecs, axis=0)
            label_embeddings.append(avg_embedding)
            valid_labels.append(label)
        else:
            # Skip labels with no embeddings
            pass
    
    if len(label_embeddings) < num_clusters:
        print(f"Warning: Only {len(label_embeddings)} verb labels have embeddings")
        print("Falling back to frequency-based clustering...")
        return cluster_by_frequent_verbs(names, verb_labels, num_clusters)
    
    label_embeddings = np.array(label_embeddings)
    print(f"Computed embeddings for {len(valid_labels)} verb labels")
    
    # Cluster verb labels using K-means
    actual_clusters = min(num_clusters, len(valid_labels))
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10, verbose=0)
    label_cluster_ids = kmeans.fit_predict(label_embeddings)
    
    # Create mapping from verb label to cluster ID
    verb_label_to_cluster = {label: int(cluster_id) 
                             for label, cluster_id in zip(valid_labels, label_cluster_ids)}
    
    # Assign default cluster for labels without embeddings
    default_cluster = 0
    
    # Assign cluster IDs to samples
    verb_cluster_labels = []
    for name in names:
        base_name = name.split('_')[0] if '_' in name else name
        verb_label = verb_labels.get(base_name, 'unknown')
        cluster_id = verb_label_to_cluster.get(verb_label, default_cluster)
        verb_cluster_labels.append(cluster_id)
    
    # Print cluster distribution
    print(f"\nSemantic verb cluster distribution:")
    unique, counts = np.unique(verb_cluster_labels, return_counts=True)
    for cluster_id, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        print(f"  Cluster {cluster_id:2d}: {count:5d} samples ({count/len(verb_cluster_labels)*100:5.1f}%)")
    
    return np.array(verb_cluster_labels), verb_label_to_cluster


def cluster_by_verb_category(names, verb_labels):
    """
    Map verbs to high-level semantic categories (manual categorization)
    
    This provides interpretable categories but requires manual verb categorization
    
    Args:
        names: list of sample names
        verb_labels: dict mapping filename to verb label
    
    Returns:
        verb_cluster_labels: (N,) array of cluster IDs
        category_to_id: dict mapping category name to cluster ID
    """
    if verb_labels is None:
        return None, None
    
    # Define semantic categories (you can extend these)
    LOCOMOTION = {'walk', 'run', 'jog', 'hop', 'skip', 'jump', 'climb', 'crawl', 'step', 
                  'march', 'stride', 'pace', 'strut', 'stroll', 'wander', 'dash', 'sprint'}
    
    UPPER_BODY = {'reach', 'grab', 'lift', 'raise', 'throw', 'wave', 'clap', 'punch',
                  'push', 'pull', 'hold', 'carry', 'pick', 'place', 'put', 'drop',
                  'catch', 'toss', 'swing', 'swipe', 'tap', 'touch', 'scratch'}
    
    LOWER_BODY = {'kick', 'squat', 'kneel', 'sit', 'stand', 'bend', 'lean', 'crouch',
                  'stoop', 'duck', 'bow', 'lunge'}
    
    DANCE_MOVEMENT = {'dance', 'sway', 'twist', 'spin', 'turn', 'rotate', 'twirl', 
                      'pivot', 'whirl'}
    
    SPORTS_EXERCISE = {'exercise', 'stretch', 'box', 'perform', 'train', 'practice',
                       'workout', 'flex', 'curl'}
    
    categories = {
        'locomotion': LOCOMOTION,
        'upper_body': UPPER_BODY,
        'lower_body': LOWER_BODY,
        'dance_movement': DANCE_MOVEMENT,
        'sports_exercise': SPORTS_EXERCISE,
        'other': set()
    }
    
    category_to_id = {cat: i for i, cat in enumerate(categories.keys())}
    
    def categorize_verbs(verb_label):
        """Assign category based on verbs present"""
        verbs = set(verb_label.split('-'))
        
        # Check each category (priority order)
        for category, verb_set in categories.items():
            if category == 'other':
                continue
            if verbs & verb_set:  # If any verb matches
                return category
        return 'other'
    
    verb_cluster_labels = []
    category_counts = defaultdict(int)
    
    for name in names:
        base_name = name.split('_')[0] if '_' in name else name
        verb_label = verb_labels.get(base_name, 'unknown')
        category = categorize_verbs(verb_label)
        cluster_id = category_to_id[category]
        verb_cluster_labels.append(cluster_id)
        category_counts[category] += 1
    
    print(f"\nCategory-based clustering ({len(categories)} categories):")
    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {category:20s}: {count:5d} samples ({count/len(verb_cluster_labels)*100:5.1f}%)")
    
    return np.array(verb_cluster_labels), category_to_id


def analyze_verb_label_statistics(verb_labels):
    """
    Print statistics about verb labels to help choose clustering strategy
    
    Args:
        verb_labels: dict mapping filename to verb label
    """
    print("\n" + "="*80)
    print("VERB LABEL STATISTICS")
    print("="*80)
    
    # Get all labels
    all_labels = list(verb_labels.values())
    label_counts = Counter(all_labels)
    
    print(f"\nTotal samples: {len(all_labels)}")
    print(f"Unique verb combinations: {len(label_counts)}")
    
    # Distribution statistics
    counts = list(label_counts.values())
    print(f"\nFrequency distribution:")
    print(f"  Mean samples per label: {np.mean(counts):.1f}")
    print(f"  Median samples per label: {np.median(counts):.1f}")
    print(f"  Max samples per label: {np.max(counts)}")
    print(f"  Min samples per label: {np.min(counts)}")
    
    # Top-K coverage analysis
    print(f"\nTop-K coverage:")
    sorted_counts = sorted(counts, reverse=True)
    total = sum(sorted_counts)
    for k in [10, 20, 50, 100]:
        if k <= len(sorted_counts):
            coverage = sum(sorted_counts[:k]) / total * 100
            print(f"  Top-{k:3d} labels cover {coverage:5.1f}% of samples")
    
    # Most common labels
    print(f"\nTop 20 most frequent verb combinations:")
    for i, (label, count) in enumerate(label_counts.most_common(20), 1):
        print(f"  {i:2d}. {label:50s}: {count:4d} samples ({count/len(all_labels)*100:4.1f}%)")
    
    # Individual verb statistics
    all_verbs = Counter()
    for label in all_labels:
        for verb in label.split('-'):
            all_verbs[verb] += 1
    
    print(f"\nTop 20 most common individual verbs:")
    for i, (verb, count) in enumerate(all_verbs.most_common(20), 1):
        print(f"  {i:2d}. {verb:20s}: {count:5d} occurrences")
    
    print(f"\nTotal unique individual verbs: {len(all_verbs)}")
    
    # Complexity distribution (number of verbs per combination)
    complexities = [len(label.split('-')) for label in all_labels]
    print(f"\nVerb combination complexity:")
    print(f"  Mean verbs per combination: {np.mean(complexities):.1f}")
    print(f"  Max verbs in a combination: {np.max(complexities)}")
    complexity_dist = Counter(complexities)
    for num_verbs, count in sorted(complexity_dist.items()):
        print(f"  {num_verbs} verb(s): {count:5d} samples ({count/len(all_labels)*100:5.1f}%)")

