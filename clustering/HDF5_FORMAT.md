# HDF5 Embedding Format

## Overview

The `extract_embeddings.py` script now saves embeddings in HDF5 format with the following hierarchical structure:

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
│   │   ├── majority_verb: "jump" (single most representative verb)
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

## Verb Extraction

Verbs are automatically extracted from the annotated text descriptions using the same logic as `dataset/humanML3d/extract_verbs.py`:

1. **Compound Verb**: All verbs found in the text, joined by hyphens and sorted alphabetically
   - Example: `"walk-run-jump"`
   
2. **Majority Verb**: The first verb (alphabetically) from the compound verb
   - Example: `"jump"` (from "walk-run-jump")
   - This provides a single label for each motion

## Usage

### Extracting Embeddings

```bash
python clustering/extract_embeddings.py \
    --dataname t2m \
    --split train \
    --resume-pth pretrained/VQVAE/net_last.pth \
    --save-dir clustering/outputs \
    --batch-size 32
```

This will generate:
- `embeddings.h5` - HDF5 file with full structure
- `encoder_embeddings.npy`, `quantized_embeddings.npy`, `code_indices.npy` - NumPy arrays (backward compatibility)
- `texts.txt`, `names.txt`, `lengths.npy` - Metadata files (backward compatibility)

### Loading HDF5 Data

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

# Load all encoder embeddings at once
def load_all_embeddings(hdf5_path):
    embeddings = []
    names = []
    majority_verbs = []
    
    with h5py.File(hdf5_path, "r") as f:
        for name in f.keys():
            embeddings.append(f[name]["encoder_embeddings"][:])
            names.append(name)
            majority_verbs.append(f[name].attrs["majority_verb"])
    
    return np.array(embeddings), names, majority_verbs
```

### Testing the HDF5 File

```bash
# Inspect file structure
python clustering/test_hdf5_loader.py \
    --hdf5-file clustering/outputs/embeddings.h5 \
    --show-samples 10

# Test loading all data
python clustering/test_hdf5_loader.py \
    --hdf5-file clustering/outputs/embeddings.h5 \
    --load-test \
    --embedding-type encoder
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
