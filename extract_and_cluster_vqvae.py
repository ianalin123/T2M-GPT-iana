import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

import models.vqvae as vqvae


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
        N, T, _ = motion_batch.shape
        
        # Preprocess: (B, T, 263) -> (B, 263, T)
        x_in = net.vqvae.preprocess(motion_batch)
        
        # Encode to continuous embeddings (B, code_dim, T')
        x_encoder = net.vqvae.encoder(x_in)

        # Continuous embeddings in (B, T', code_dim) for clustering
        x_encoder_post = net.vqvae.postprocess(x_encoder)

        # Flatten for discrete index lookup (matches VQVAE.encode)
        x_encoder_flat = x_encoder_post.contiguous().view(-1, x_encoder_post.shape[-1])  # (B*T', code_dim)

        # Quantize using original (B, code_dim, T') tensor
        quantized, loss, perplexity = net.vqvae.quantizer(x_encoder)

        # Postprocess quantized embeddings to (B, T', code_dim)
        quantized_post = net.vqvae.postprocess(quantized)

        # Get discrete indices
        code_indices = net.vqvae.quantizer.quantize(x_encoder_flat)
        code_indices = code_indices.view(N, -1)
        
    return {
        'encoder_embeddings': x_encoder_post.cpu().numpy(),  # CONTINUOUS embeddings for clustering
        'quantized_embeddings': quantized_post.cpu().numpy(),
        'code_indices': code_indices.cpu().numpy(),
        'loss': loss.item() if hasattr(loss, 'item') else loss,
        'perplexity': perplexity.item() if hasattr(perplexity, 'item') else perplexity
    }


def load_dataset(args, split='train'):
    """Load motion dataset
    
    Note: The eval dataset returns raw motion data which is needed for VQ-VAE encoding.
    For train split, we use val split as a workaround (train dataset returns tokenized motions).
    """
    from dataset import dataset_TM_eval
    from utils.word_vectorizer import WordVectorizer
    
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    
    # Dataset_TM_eval only supports val/test splits (not train)
    # For train, we'll use val as a workaround since train dataset returns tokenized data
    if split == 'train':
        print("Warning: Train split not directly supported. Using val split instead.")
        print("         (Train dataset returns tokenized motions, not raw motion vectors)")
        is_test = False  # Use val split
    else:
        is_test = (split == 'test')
    
    dataset = dataset_TM_eval.Text2MotionDataset(
        args.dataname,
        is_test=is_test,
        w_vectorizer=w_vectorizer,
        feat_bias=args.feat_bias,
        unit_length=args.unit_length if hasattr(args, 'unit_length') else 4
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    return dataset, dataloader


def extract_all_embeddings(net, dataloader, device='cuda', max_batches=None):
    """
    Extract embeddings from entire dataset
    
    Returns:
        embeddings_dict: dictionary with all extracted data
    """
    all_encoder_embeddings = []
    all_quantized_embeddings = []
    all_code_indices = []
    all_motions = []
    all_texts = []
    all_lengths = []
    
    print("Extracting embeddings from dataset...")
    
    for i, batch in enumerate(tqdm(dataloader)):
        if max_batches and i >= max_batches:
            break
        
        # Unpack batch tuple: word_embeddings, pos_one_hots, text/caption, sent_len, motion, motion_length, token, name
        word_embeddings, pos_one_hots, text, sent_len, motion, motion_length, token, name = batch
        
        motion = motion.to(device)  # (B, T, 263)
        
        # Extract embeddings
        result = extract_encoder_embeddings(net, motion)
        
        all_encoder_embeddings.append(result['encoder_embeddings'])
        all_quantized_embeddings.append(result['quantized_embeddings'])
        all_code_indices.append(result['code_indices'])
        all_motions.append(motion.cpu().numpy())
        all_texts.extend(text)
        all_lengths.extend(motion_length.cpu().numpy())
    
    # Concatenate all batches
    return {
        'encoder_embeddings': np.concatenate(all_encoder_embeddings, axis=0),  # (N, T', D)
        'quantized_embeddings': np.concatenate(all_quantized_embeddings, axis=0),
        'code_indices': np.concatenate(all_code_indices, axis=0),  # (N, T')
        'motions': np.concatenate(all_motions, axis=0),  # (N, T, 263)
        'texts': all_texts,
        'lengths': np.array(all_lengths)
    }


def cluster_embeddings(embeddings, n_clusters=20, aggregate='mean', pca_dim=50):
    """
    Cluster the continuous encoder embeddings
    
    Args:
        embeddings: (N, T', D) array of encoder embeddings
        n_clusters: number of clusters
        aggregate: 'mean', 'max', or None
        pca_dim: dimensionality reduction (None to skip)
    
    Returns:
        labels: (N,) cluster assignments
        kmeans: fitted KMeans model
        embeddings_agg: (N, D) aggregated embeddings used for clustering
    """
    print(f"\nOriginal embeddings shape: {embeddings.shape}")
    
    # Aggregate over time dimension
    if aggregate == 'mean':
        embeddings_agg = embeddings.mean(axis=1)  # (N, D)
        print(f"Aggregated with mean pooling: {embeddings_agg.shape}")
    elif aggregate == 'max':
        embeddings_agg = embeddings.max(axis=1)
        print(f"Aggregated with max pooling: {embeddings_agg.shape}")
    elif aggregate == 'flatten':
        embeddings_agg = embeddings.reshape(embeddings.shape[0], -1)
        print(f"Flattened: {embeddings_agg.shape}")
    else:
        embeddings_agg = embeddings
    
    # PCA dimensionality reduction
    if pca_dim and embeddings_agg.shape[1] > pca_dim:
        print(f"\nReducing dimensionality with PCA to {pca_dim}...")
        pca = PCA(n_components=pca_dim)
        embeddings_agg = pca.fit_transform(embeddings_agg)
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    else:
        pca = None
    
    # K-means clustering
    print(f"\nClustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1)
    labels = kmeans.fit_predict(embeddings_agg)
    
    # Print cluster distribution
    print("\nCluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  Cluster {label:2d}: {count:5d} samples ({count/len(labels)*100:5.1f}%)")
    
    return labels, kmeans, embeddings_agg, pca


def visualize_clusters(embeddings, labels, texts, save_dir, max_samples=5000):
    """Create PCA, t-SNE and UMAP visualizations"""
    
    # Subsample if too many
    if len(embeddings) > max_samples:
        print(f"\nSubsampling {max_samples} points for visualization...")
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings_vis = embeddings[indices]
        labels_vis = labels[indices]
    else:
        embeddings_vis = embeddings
        labels_vis = labels
    
    # PCA - Fast and interpretable!
    print("\nComputing PCA...")
    pca_viz = PCA(n_components=2)
    embeddings_pca = pca_viz.fit_transform(embeddings_vis)
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1],
                         c=labels_vis, cmap='tab20', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'PCA Visualization of Latent Embeddings\n'
              f'(PC1: {pca_viz.explained_variance_ratio_[0]:.1%}, '
              f'PC2: {pca_viz.explained_variance_ratio_[1]:.1%}, '
              f'Total: {pca_viz.explained_variance_ratio_.sum():.1%})')
    plt.xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]:.1%} variance)')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/clusters_pca.png', dpi=300, bbox_inches='tight')
    print(f"Saved PCA visualization to {save_dir}/clusters_pca.png")
    print(f"  PC1 explains {pca_viz.explained_variance_ratio_[0]:.1%} of variance")
    print(f"  PC2 explains {pca_viz.explained_variance_ratio_[1]:.1%} of variance")
    plt.close()
    
    # PCA with more components for scree plot
    print("\nComputing PCA scree plot...")
    pca_full = PCA(n_components=min(50, embeddings_vis.shape[1]))
    pca_full.fit(embeddings_vis)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
             pca_full.explained_variance_ratio_, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum) + 1), cumsum, 'ro-')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Variance Explained')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pca_variance.png', dpi=300, bbox_inches='tight')
    print(f"Saved PCA variance analysis to {save_dir}/pca_variance.png")
    plt.close()
    
    # t-SNE (slower but good for local structure)
    print("\nComputing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=1)
    embeddings_tsne = tsne.fit_transform(embeddings_vis)
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                         c=labels_vis, cmap='tab20', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Cluster')
    plt.title('t-SNE Visualization of Latent Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/clusters_tsne.png', dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE visualization to {save_dir}/clusters_tsne.png")
    plt.close()
    


def analyze_clusters(labels, texts, code_indices, save_dir, top_k=10):
    """Analyze what each cluster represents"""
    
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS")
    print("="*80)
    
    n_clusters = len(np.unique(labels))
    
    analysis_file = open(f'{save_dir}/cluster_analysis.txt', 'w')
    
    for cluster_id in range(n_clusters):
        mask = (labels == cluster_id)
        cluster_texts = [texts[i] for i, m in enumerate(mask) if m]
        cluster_codes = code_indices[mask]
        
        analysis = f"\n{'='*80}\n"
        analysis += f"Cluster {cluster_id} - {len(cluster_texts)} samples ({len(cluster_texts)/len(labels)*100:.1f}%)\n"
        analysis += f"{'='*80}\n"
        
        # Sample text descriptions
        analysis += f"\nSample descriptions (first {top_k}):\n"
        for i, text in enumerate(cluster_texts[:top_k]):
            analysis += f"  {i+1}. {text}\n"
        
        # Most common discrete codes used
        all_codes = cluster_codes.flatten()
        unique_codes, counts = np.unique(all_codes, return_counts=True)
        top_code_indices = np.argsort(-counts)[:10]
        
        analysis += f"\nMost frequently used codebook entries:\n"
        for idx in top_code_indices:
            code = unique_codes[idx]
            count = counts[idx]
            analysis += f"  Code {code:3d}: used {count:5d} times ({count/len(all_codes)*100:.1f}%)\n"
        
        analysis += f"\nCodebook diversity: {len(unique_codes)}/{code_indices.max()+1} codes used\n"
        
        print(analysis)
        analysis_file.write(analysis)
    
    analysis_file.close()
    print(f"\nSaved full analysis to {save_dir}/cluster_analysis.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='t2m', type=str, help='dataset name')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--resume-pth', default='pretrained/VQVAE/net_last.pth', 
                       type=str, help='VQ-VAE checkpoint path')
    parser.add_argument('--n-clusters', default=20, type=int)
    parser.add_argument('--save-dir', default='./cluster_analysis', type=str)
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--max-batches', default=None, type=int,
                       help='Limit number of batches for quick testing')
    
    # VQ-VAE architecture params (must match training)
    parser.add_argument('--nb-code', default=512, type=int)
    parser.add_argument('--code-dim', default=512, type=int)
    parser.add_argument('--output-emb-width', default=512, type=int)
    parser.add_argument('--down-t', default=2, type=int)
    parser.add_argument('--stride-t', default=2, type=int)
    parser.add_argument('--width', default=512, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--dilation-growth-rate', default=3, type=int)
    parser.add_argument('--vq-act', default='relu', type=str)
    parser.add_argument('--quantizer', default='ema_reset', type=str, 
                       choices=['ema', 'orig', 'ema_reset', 'reset'])
    
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load VQ-VAE model
    print("="*80)
    print("Loading VQ-VAE model...")
    print("="*80)
    
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
        activation=args.vq_act
    )
    
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    
    # Handle device selection (CUDA, MPS for Mac, or CPU)
    if torch.cuda.is_available():
        device = 'cuda'
        net.cuda()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        net.to('mps')
    else:
        device = 'cpu'
        net.to('cpu')
    print(f"Using device: {device}")
    print(f"Loaded checkpoint from {args.resume_pth}")
    print(f"Codebook size: {args.nb_code}, Code dimension: {args.code_dim}")
    
    # Load dataset
    print("\n" + "="*80)
    print("Loading dataset...")
    print("="*80)
    
    # Set required args for dataset
    if not hasattr(args, 'feat_bias'):
        args.feat_bias = 5
    if not hasattr(args, 'unit_length'):
        args.unit_length = 4
    
    dataset, dataloader = load_dataset(args, split=args.split)
    print(f"Dataset: {args.dataname}, Split: {args.split}, Size: {len(dataset)}")
    
    # Extract embeddings
    print("\n" + "="*80)
    print("Extracting embeddings...")
    print("="*80)
    
    data = extract_all_embeddings(net, dataloader, device=device, max_batches=args.max_batches)
    
    print(f"\nExtracted data shapes:")
    print(f"  Encoder embeddings: {data['encoder_embeddings'].shape}")
    print(f"  Code indices: {data['code_indices'].shape}")
    print(f"  Number of texts: {len(data['texts'])}")
    
    # Save raw data
    print(f"\nSaving raw embeddings to {args.save_dir}/...")
    np.save(f'{args.save_dir}/encoder_embeddings.npy', data['encoder_embeddings'])
    np.save(f'{args.save_dir}/code_indices.npy', data['code_indices'])
    np.save(f'{args.save_dir}/motions.npy', data['motions'])
    with open(f'{args.save_dir}/texts.txt', 'w') as f:
        f.write('\n'.join(data['texts']))
    
    # Cluster embeddings
    print("\n" + "="*80)
    print("Clustering embeddings...")
    print("="*80)
    
    labels, kmeans, embeddings_processed, pca = cluster_embeddings(
        data['encoder_embeddings'],
        n_clusters=args.n_clusters,
        aggregate='mean',
        pca_dim=50
    )
    
    # Save clustering results
    np.save(f'{args.save_dir}/cluster_labels.npy', labels)
    np.save(f'{args.save_dir}/embeddings_processed.npy', embeddings_processed)
    
    # Visualize
    print("\n" + "="*80)
    print("Visualizing clusters...")
    print("="*80)
    visualize_clusters(embeddings_processed, labels, data['texts'], args.save_dir)
    
    # Analyze
    print("\n" + "="*80)
    print("Analyzing clusters...")
    print("="*80)
    analyze_clusters(labels, data['texts'], data['code_indices'], args.save_dir)
    
    print("\n" + "="*80)
    print(f"✓ Done! All results saved to {args.save_dir}/")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - encoder_embeddings.npy: continuous latent embeddings (N, T', D)")
    print(f"  - code_indices.npy: discrete code indices (N, T')")
    print(f"  - cluster_labels.npy: cluster assignments (N,)")
    print(f"  - embeddings_processed.npy: processed embeddings used for clustering")
    print(f"  - texts.txt: motion text descriptions")
    print(f"  - clusters_tsne.png: t-SNE visualization")
    print(f"  - cluster_analysis.txt: detailed cluster analysis")


if __name__ == '__main__':
    main()