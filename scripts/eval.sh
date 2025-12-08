

# CUDA_VISIBLE_DEVICES=3 python VQ_eval.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir /scratch/2025_10/malulekevon/motion_latent_space/output_vq/ --dataname t2m --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name t2m_vq_512 --resume-pth pretrained/VQVAE/net_last.pth 


#/scratch/2025_10/malulekevon/t2m_gpt/output/VQVAE_512_512_t2m_20251130_055053/models/best_model.pth
# /scratch/2025_10/malulekevon/t2m_gpt/output/VQVAE_512_512_kit_20251130_054524/models/best_model.pth


#t2m
# CUDA_VISIBLE_DEVICES=3 python VQ_eval.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 128 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir /scratch/2025_10/malulekevon/motion_latent_space/output_vq/ --dataname t2m --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name t2m_vq_128 --resume-pth /scratch/2025_10/malulekevon/t2m_gpt/output/VQVAE_128_512_t2m_20251130_055053/models/best_model.pth

# CUDA_VISIBLE_DEVICES=3 python VQ_eval.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 256 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir /scratch/2025_10/malulekevon/motion_latent_space/output_vq/ --dataname t2m --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name t2m_vq_256 --resume-pth /scratch/2025_10/malulekevon/t2m_gpt/output/VQVAE_256_512_t2m_20251130_055053/models/best_model.pth

 CUDA_VISIBLE_DEVICES=3 python VQ_eval.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir /scratch/2025_10/malulekevon/motion_latent_space/output_vq/ --dataname t2m --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name t2m_vq_512_ours --resume-pth /scratch/2025_10/malulekevon/t2m_gpt/output/VQVAE_512_512_t2m_20251130_055053/models/best_model.pth

# CUDA_VISIBLE_DEVICES=3 python VQ_eval.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 1024 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir /scratch/2025_10/malulekevon/motion_latent_space/output_vq/ --dataname t2m --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name t2m_vq_1024 --resume-pth /scratch/2025_10/malulekevon/t2m_gpt/output/VQVAE_1024_512_t2m_20251130_055052/models/best_model.pth


####

#kit
CUDA_VISIBLE_DEVICES=3 python VQ_eval.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 128 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir /scratch/2025_10/malulekevon/motion_latent_space/output_vq/ --dataname kit --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name kit_vq_128 --resume-pth /scratch/2025_10/malulekevon/t2m_gpt/output/VQVAE_128_512_kit_20251130_054524/models/best_model.pth


CUDA_VISIBLE_DEVICES=3 python VQ_eval.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 256 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir /scratch/2025_10/malulekevon/motion_latent_space/output_vq/ --dataname kit --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name kit_vq_256 --resume-pth /scratch/2025_10/malulekevon/t2m_gpt/output/VQVAE_256_512_kit_20251130_054524/models/best_model.pth

CUDA_VISIBLE_DEVICES=3 python VQ_eval.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 1024 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir /scratch/2025_10/malulekevon/motion_latent_space/output_vq/ --dataname kit --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name kit_vq_1024 --resume-pth /scratch/2025_10/malulekevon/t2m_gpt/output/VQVAE_1024_512_kit_20251130_054524/models/best_model.pth


CUDA_VISIBLE_DEVICES=3 python VQ_eval.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir /scratch/2025_10/malulekevon/motion_latent_space/output_vq/ --dataname kit --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name kit_vq_512 --resume-pth /scratch/2025_10/malulekevon/t2m_gpt/output/VQVAE_512_512_kit_20251130_054524/models/best_model.pth