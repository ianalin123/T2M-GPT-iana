import os
import sys
import argparse
from collections import Counter

import h5py
import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is on the path so we can import models and dataset
# Since we're now in clustering/, go up one level to get to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import models.vqvae as vqvae  # noqa: E402


def load_verb_labels_from_file(verb_file_path):
    """
    Load pre-extracted verb labels from verbs.txt file.

    Args:
        verb_file_path: Path to verbs.txt file (e.g., dataset/HumanML3D/verbs.txt)

    Returns:
        dict mapping file_id to compound verb string
    """
    verb_labels = {}

    if not os.path.exists(verb_file_path):
        print(f"Warning: Verb labels file not found at {verb_file_path}")
        return verb_labels

    with open(verb_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                file_id, compound_verb = parts
                verb_labels[file_id] = compound_verb
            else:
                # Handle case where there might be no verbs
                verb_labels[parts[0]] = ""

    print(f"Loaded {len(verb_labels)} verb labels from {verb_file_path}")
    return verb_labels




def get_majority_verb(compound_verb):
    """
    Get the majority (most common) verb from a compound verb string.

    Args:
        compound_verb: String like "walk-run-jump" (alphabetically sorted verbs)

    Returns:
        Single most common verb, or first alphabetically if tie
    """
    if not compound_verb or compound_verb == "":
        return ""

    verbs = compound_verb.split("-")
    if len(verbs) == 1:
        return verbs[0]

    # For compound verbs, return the first one (they're already sorted alphabetically)
    # In future could use frequency analysis from training data
    return verbs[0]


def extract_verb_labels(names, verb_labels_dict):
    """
    Extract compound and majority verb labels for samples using pre-extracted verb labels.

    Args:
        names: List of sample names/IDs to map to verb labels
        verb_labels_dict: dict mapping file_id to compound verb string (from verbs.txt)

    Returns:
        dict with:
            - compound_verbs: list of compound verb strings (e.g., "walk-run")
            - majority_verbs: list of single majority verbs
    """
    compound_verbs = []
    majority_verbs = []

    for name in names:
        # Remove any prefix (e.g., "A_" in "A_000021" -> "000021")
        # Some samples have letter prefixes for temporal splits
        base_name = name.split('_')[-1] if '_' in name else name

        # Get verb label from dictionary
        compound_verb = verb_labels_dict.get(base_name, "")
        majority_verb = get_majority_verb(compound_verb)

        compound_verbs.append(compound_verb)
        majority_verbs.append(majority_verb)

    return {
        "compound_verbs": compound_verbs,
        "majority_verbs": majority_verbs,
    }


def extract_encoder_embeddings(net, motion_batch):
    """
    Extract continuous encoder outputs BEFORE quantization

    Args:
        net: HumanVQVAE model
        motion_batch: (B, T, 263) motion sequences

    Returns:
        dict with:
            - encoder_embeddings: (B, T', code_dim) continuous pre-quantization
            - quantized_embeddings: (B, T', code_dim) post-quantization
            - code_indices: (B, T') discrete code indices
    """
    net.eval()
    with torch.no_grad():
        N, _, _ = motion_batch.shape

        # Preprocess: (B, T, 263) -> (B, 263, T)
        x_in = net.vqvae.preprocess(motion_batch)

        # Encode to continuous embeddings (B, code_dim, T')
        x_encoder = net.vqvae.encoder(x_in)

        # Continuous embeddings in (B, T', code_dim) for clustering / analysis
        x_encoder_post = net.vqvae.postprocess(x_encoder)

        # Flatten for discrete index lookup (matches VQVAE.encode)
        x_encoder_flat = x_encoder_post.contiguous().view(-1, x_encoder_post.shape[-1])

        # Quantize using original (B, code_dim, T') tensor
        quantized, loss, perplexity = net.vqvae.quantizer(x_encoder)

        # Postprocess quantized embeddings to (B, T', code_dim)
        quantized_post = net.vqvae.postprocess(quantized)

        # Get discrete indices
        code_indices = net.vqvae.quantizer.quantize(x_encoder_flat)
        code_indices = code_indices.view(N, -1)

    return {
        "encoder_embeddings": x_encoder_post.cpu().numpy(),  # continuous embeddings
        "quantized_embeddings": quantized_post.cpu().numpy(),
        "code_indices": code_indices.cpu().numpy(),
        "loss": loss.item() if hasattr(loss, "item") else loss,
        "perplexity": perplexity.item() if hasattr(perplexity, "item") else perplexity,
    }


def load_dataset(args, split="train"):
    """
    Load motion dataset.

    Note: The eval dataset returns raw motion data which is needed for VQ-VAE encoding.
    For train split, we use val split as a workaround (train dataset returns tokenized motions).
    """
    from dataset import dataset_TM_eval
    from utils.word_vectorizer import WordVectorizer

    w_vectorizer = WordVectorizer("./glove", "our_vab")

    # Dataset_TM_eval only supports val/test splits (not train)
    # For train, we'll use val as a workaround since train dataset returns tokenized data
    if split == "train":
        print("Warning: Train split not directly supported. Using val split instead.")
        print("         (Train dataset returns tokenized motions, not raw motion vectors)")
        is_test = False  # Use val split
    else:
        is_test = split == "test"

    dataset = dataset_TM_eval.Text2MotionDataset(
        args.dataname,
        is_test=is_test,
        w_vectorizer=w_vectorizer,
        feat_bias=args.feat_bias,
        unit_length=args.unit_length if hasattr(args, "unit_length") else 4,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    return dataset, dataloader


def extract_all_embeddings(net, dataloader, device="cuda", max_batches=None):
    """
    Extract embeddings from entire dataset.

    Returns:
        dict with:
            encoder_embeddings: (N, T', D)
            quantized_embeddings: (N, T', D)
            code_indices: (N, T')
            motions: (N, T, 263)
            texts: list[str]
            lengths: (N,)
            names: list[str]
    """
    all_encoder_embeddings = []
    all_quantized_embeddings = []
    all_code_indices = []
    all_motions = []
    all_texts = []
    all_lengths = []
    all_names = []

    print("Extracting embeddings from dataset...")

    for i, batch in enumerate(tqdm(dataloader)):
        if max_batches and i >= max_batches:
            break

        # Unpack batch tuple: word_embeddings, pos_one_hots, text/caption,
        # sent_len, motion, motion_length, token, name
        (
            word_embeddings,
            pos_one_hots,
            text,
            sent_len,
            motion,
            motion_length,
            token,
            name,
        ) = batch

        motion = motion.to(device)  # (B, T, 263)

        # Extract embeddings
        result = extract_encoder_embeddings(net, motion)

        all_encoder_embeddings.append(result["encoder_embeddings"])
        all_quantized_embeddings.append(result["quantized_embeddings"])
        all_code_indices.append(result["code_indices"])
        all_motions.append(motion.cpu().numpy())
        all_texts.extend(text)
        all_lengths.extend(motion_length.cpu().numpy())
        all_names.extend(name)  # Store names for verb label mapping

    # Concatenate all batches
    return {
        "encoder_embeddings": np.concatenate(all_encoder_embeddings, axis=0),
        "quantized_embeddings": np.concatenate(all_quantized_embeddings, axis=0),
        "code_indices": np.concatenate(all_code_indices, axis=0),
        "motions": np.concatenate(all_motions, axis=0),
        "texts": all_texts,
        "lengths": np.array(all_lengths),
        "names": all_names,
    }


def save_embeddings_hdf5(data, save_path, split="train", verb_labels_dict=None):
    """
    Save embeddings and metadata to HDF5 file with hierarchical structure.

    Structure:
        /[file_id]/
            compound_verb (attribute)
            majority_verb (attribute)
            text (attribute)
            length (attribute)
            encoder_embeddings (dataset)
            quantized_embeddings (dataset)
            code_indices (dataset)

    Args:
        data: dict with encoder_embeddings, quantized_embeddings, code_indices, texts, names, etc.
        save_path: Path to save HDF5 file
        split: Dataset split name (train/val/test)
        verb_labels_dict: dict mapping file_id to compound verb string (loaded from verbs.txt)
                         Format: {"000021": "walk", "M009233": "stop-walk", ...}
    """
    print(f"\nSaving embeddings to HDF5: {save_path}")

    if verb_labels_dict is None:
        raise ValueError("verb_labels_dict is required. Please run extract_verbs.py first to generate verbs.txt")

    # Map verb labels to samples using their IDs
    print("Mapping verb labels to samples...")
    verb_labels = extract_verb_labels(data["names"], verb_labels_dict)

    # Report statistics
    non_empty_verbs = sum(1 for v in verb_labels["compound_verbs"] if v)
    print(f"  Successfully mapped {non_empty_verbs}/{len(data['names'])} samples to verb labels")
    if non_empty_verbs < len(data["names"]):
        missing = len(data["names"]) - non_empty_verbs
        print(f"  Warning: {missing} samples have no verb labels in verbs.txt")

    with h5py.File(save_path, "w") as f:
        # Add metadata attributes at root level
        f.attrs["split"] = split
        f.attrs["n_samples"] = len(data["names"])
        f.attrs["embedding_dim"] = data["encoder_embeddings"].shape[-1]

        # Create a group for each sample
        for idx, name in enumerate(tqdm(data["names"], desc="Writing to HDF5")):
            # Create group for this sample using file ID as key
            sample_group = f.create_group(name)

            # Add verb labels as attributes
            sample_group.attrs["compound_verb"] = verb_labels["compound_verbs"][idx]
            sample_group.attrs["majority_verb"] = verb_labels["majority_verbs"][idx]

            # Add embeddings as datasets
            sample_group.create_dataset(
                "encoder_embeddings",
                data=data["encoder_embeddings"][idx],
                compression="gzip",
                compression_opts=4,
            )
            sample_group.create_dataset(
                "quantized_embeddings",
                data=data["quantized_embeddings"][idx],
                compression="gzip",
                compression_opts=4,
            )
            sample_group.create_dataset(
                "code_indices",
                data=data["code_indices"][idx],
                compression="gzip",
                compression_opts=4,
            )

            # Add metadata as attributes
            sample_group.attrs["text"] = data["texts"][idx]
            sample_group.attrs["length"] = int(data["lengths"][idx])

    print(f"✓ Saved {len(data['names'])} samples to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract VQ-VAE encoder embeddings from the motion dataset"
    )
    parser.add_argument("--dataname", default="t2m", type=str, help="dataset name")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument(
        "--resume-pth",
        default="pretrained/VQVAE/net_last.pth",
        type=str,
        help="VQ-VAE checkpoint path",
    )
    parser.add_argument(
        "--save-dir",
        default="clustering/outputs",
        type=str,
        help="Directory to save extracted embeddings and metadata",
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "val", "test"], help="data split"
    )
    parser.add_argument(
        "--max-batches",
        default=None,
        type=int,
        help="Limit number of batches for quick testing",
    )

    # VQ-VAE architecture params (must match training)
    parser.add_argument("--nb-code", default=512, type=int)
    parser.add_argument("--code-dim", default=512, type=int)
    parser.add_argument("--output-emb-width", default=512, type=int)
    parser.add_argument("--down-t", default=2, type=int)
    parser.add_argument("--stride-t", default=2, type=int)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--dilation-growth-rate", default=3, type=int)
    parser.add_argument("--vq-act", default="relu", type=str)
    parser.add_argument(
        "--quantizer",
        default="ema_reset",
        type=str,
        choices=["ema", "orig", "ema_reset", "reset"],
    )
    parser.add_argument(
        "--mu",
        default=0.99,
        type=float,
        help="exponential moving average to update the codebook",
    )

    args = parser.parse_args()

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Load VQ-VAE model
    print("=" * 80)
    print("Loading VQ-VAE model...")
    print("=" * 80)

    net = vqvae.HumanVQVAE(
        args,
        args.nb_code,
        args.code_dim,
        args.output_emb_width,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate,
        activation=args.vq_act,
    )

    ckpt = torch.load(args.resume_pth, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    net.eval()

    # Handle device selection (CUDA, MPS for Mac, or CPU)
    if torch.cuda.is_available():
        device = "cuda"
        net.cuda()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        net.to("mps")
    else:
        device = "cpu"
        net.to("cpu")
    print(f"Using device: {device}")
    print(f"Loaded checkpoint from {args.resume_pth}")
    print(f"Codebook size: {args.nb_code}, Code dimension: {args.code_dim}")

    # Load dataset
    print("\n" + "=" * 80)
    print("Loading dataset...")
    print("=" * 80)

    # Set required args for dataset
    if not hasattr(args, "feat_bias"):
        args.feat_bias = 5
    if not hasattr(args, "unit_length"):
        args.unit_length = 4

    dataset, dataloader = load_dataset(args, split=args.split)
    print(f"Dataset: {args.dataname}, Split: {args.split}, Size: {len(dataset)}")

    # Load verb labels from file
    verb_labels_dict = None
    if args.dataname == 't2m':
        verb_file_path = './dataset/HumanML3D/verbs.txt'
        verb_labels_dict = load_verb_labels_from_file(verb_file_path)
    elif args.dataname == 'kit':
        verb_file_path = './dataset/KIT-ML/verbs.txt'
        if os.path.exists(verb_file_path):
            verb_labels_dict = load_verb_labels_from_file(verb_file_path)
        else:
            print(f"Warning: No verb labels file found for KIT dataset at {verb_file_path}")

    # Extract embeddings
    print("\n" + "=" * 80)
    print("Extracting embeddings...")
    print("=" * 80)

    data = extract_all_embeddings(
        net, dataloader, device=device, max_batches=args.max_batches
    )

    print("\nExtracted data shapes:")
    print(f"  Encoder embeddings: {data['encoder_embeddings'].shape}")
    print(f"  Quantized embeddings: {data['quantized_embeddings'].shape}")
    print(f"  Code indices: {data['code_indices'].shape}")
    print(f"  Number of texts: {len(data['texts'])}")
    print(f"  Number of samples: {len(data['names'])}")

    # Save to HDF5
    print("\n" + "=" * 80)
    print("Saving embeddings to HDF5...")
    print("=" * 80)

    # Save to HDF5 format (with verbs)
    hdf5_path = os.path.join(args.save_dir, "embeddings.h5")
    save_embeddings_hdf5(data, hdf5_path, split=args.split, verb_labels_dict=verb_labels_dict)

    # Also save numpy files for backward compatibility
    print(f"\nSaving numpy files for backward compatibility to {args.save_dir}/...")
    np.save(
        os.path.join(args.save_dir, "encoder_embeddings.npy"),
        data["encoder_embeddings"],
    )
    np.save(
        os.path.join(args.save_dir, "quantized_embeddings.npy"),
        data["quantized_embeddings"],
    )
    np.save(os.path.join(args.save_dir, "code_indices.npy"), data["code_indices"])
    np.save(os.path.join(args.save_dir, "lengths.npy"), data["lengths"])
    with open(os.path.join(args.save_dir, "texts.txt"), "w") as f:
        f.write("\n".join(data["texts"]))
    with open(os.path.join(args.save_dir, "names.txt"), "w") as f:
        f.write("\n".join(data["names"]))

    print("\n" + "=" * 80)
    print(f"✓ Done! Embeddings saved to {hdf5_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()


