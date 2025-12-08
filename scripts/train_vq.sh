conda activate T2M-GPT

dataset=t2m
batch_size=256
#Original model config

CUDA_VISIBLE_DEVICES=3 python train_vq.py \
--batch-size $batch_size \
--lr 2e-4 \
--total-iter 300000 \
--warm-up-iter 1000 \
--lr-scheduler 200000 \
--nb-code 512 \
--code-dim 512 \
--width 512 \
--output-emb-width 512 \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir output \
--dataname $dataset \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name VQVAE_512_512 &


#128 codebook size
CUDA_VISIBLE_DEVICES=5 python train_vq.py \
--batch-size $batch_size \
--lr 2e-4 \
--total-iter 300000 \
--warm-up-iter 1000 \
--lr-scheduler 200000 \
--nb-code 128 \
--code-dim 512 \
--width 512 \
--output-emb-width 512 \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir output \
--dataname $dataset \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name VQVAE_128_512  &


#256 codebook size
CUDA_VISIBLE_DEVICES=6 python train_vq.py \
--batch-size $batch_size \
--lr 2e-4 \
--total-iter 300000 \
--warm-up-iter 1000 \
--lr-scheduler 200000 \
--nb-code 256 \
--code-dim 512 \
--width 512 \
--output-emb-width 512 \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir output \
--dataname $dataset \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name VQVAE_256_512 &


#1024 codebook size
CUDA_VISIBLE_DEVICES=7 python train_vq.py \
--batch-size $batch_size \
--lr 2e-4 \
--total-iter 300000 \
--warm-up-iter 1000 \
--lr-scheduler 200000 \
--nb-code 1024 \
--code-dim 512 \
--width 512 \
--output-emb-width 512 \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir output \
--dataname $dataset \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name VQVAE_1024_512 &