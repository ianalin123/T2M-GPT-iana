# HDF5 Embedding Format

## Overview

The `extract_embeddings.py` script saves embeddings in HDF5 format with the following hierarchical structure:

```
embeddings.h5
├── [Root Attributes]
│   ├── split: "train" | "val" | "test"
│   ├── n_samples: number of samples
│   └── embedding_dim: dimensionality of embeddings
│
├── [file_id_1]/  (e.g., "000001", "M012345")
│   ├── [Attributes]
│   │   ├── compound_verb: "walk-run-jump" (all verbs, sorted alphabetically)

│   │   ├── text: full text description with POS tags
│   │   └── length: sequence length
│   │
│   └── [Datasets]
│       ├── encoder_embeddings: (T', D) continuous pre-quantization embeddings
│       ├── quantized_embeddings: (T', D) post-quantization embeddings
│       └── code_indices: (T',) discrete codebook indices
│
├── [file_id_2]/
│   └── ...
...
```

### Checking HDF5 Data

```python
import h5py
import numpy as np

# Open file
with h5py.File("clustering/outputs/embeddings.h5", "r") as f:
    # Access a specific sample
    sample = f["000001"]
    
    # Get embeddings
    encoder_emb = sample["encoder_embeddings"][:]  # (T', D)
    quantized_emb = sample["quantized_embeddings"][:]  # (T', D)
    code_indices = sample["code_indices"][:]  # (T',)
    
    # Get verb labels
    compound_verb = sample.attrs["compound_verb"]  # e.g., "walk-run"
    majority_verb = sample.attrs["majority_verb"]  # e.g., "run"
    
    # Get text and metadata
    text = sample.attrs["text"]
    length = sample.attrs["length"]

```
## After clustering:
The `run_clustering.py` script can optionally write a *clustered* HDF5 file with assignments and verb labels:
```
embeddings_clustered.h5
├── [Root Attributes]
│ ├── n_samples: number of samples
│ ├── n_clusters: number of K-means clusters
│ ├── embedding_dim: dimensionality of encoder embeddings
│ ├── cluster_0_label: atomic verb label for cluster 0 (e.g., "walk")
│ ├── cluster_1_label: atomic verb label for cluster 1
│ ├── ...
│ └── cluster_{K-1}label: atomic verb label for cluster K-1
│
├── [file_id_1]/ (e.g., "000001", "M012345")
│ ├── [Attributes]
│ │ ├── compound_verb: original compound verb from embeddings.h5 / verbs.txt
│ │ ├── cluster_id: integer K-means cluster assignment
│ │ ├── cluster_label: atomic verb label for this cluster (e.g., "run")
│ │ ├── text: full text description with POS tags
│ │ └── length: sequence length (derived from encoder embeddings shape)
│ │
│ └── [Datasets]
│ ├── encoder_embeddings: (T', D) continuous pre-quantization embeddings
│ └── code_indices: (T',) discrete codebook indices
│
├── [file_id_2]/
│ └── ...
└── ...
```

## Backward Compatibility

The script still outputs the old NumPy format files to maintain compatibility with `run_clustering.py`:
- `encoder_embeddings.npy`
- `quantized_embeddings.npy`
- `code_indices.npy`
- `texts.txt`
- `names.txt`
- `lengths.npy`

## Benefits of HDF5 Format

1. **Hierarchical Organization**: Each sample is stored in its own group with all related data
2. **Verb Labels Built-in**: No need to separately load `verbs.txt` file
3. **Efficient Storage**: Compression reduces file size
4. **Selective Loading**: Load only specific samples or fields without reading entire file
5. **Self-Documenting**: Structure includes metadata and is easily inspectable with `h5py` or HDFView
