import os
import json
import sys
import signal

# Clean LD_LIBRARY_PATH to avoid conflicts
if 'LD_LIBRARY_PATH' in os.environ:
    # Keep only essential CUDA paths
    old_ld = os.environ['LD_LIBRARY_PATH']
    # Filter out conflicting paths
    clean_paths = [p for p in old_ld.split(':') if 'cuda' in p.lower() or 'nvidia' in p.lower()]
    os.environ['LD_LIBRARY_PATH'] = ':'.join(clean_paths) if clean_paths else ''

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


import models.vqvae as vqvae
import utils.losses as losses 
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_VQ, dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from utils.word_vectorizer import WordVectorizer
import wandb
from tqdm import tqdm
import yaml
import argparse
from datetime import datetime

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

##### ---- Exp dirs ---- #####
train_mode = False

args = option_vq.get_args_parser()

if args.config != 'None' and train_mode:
  with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

  for key, value in config.items():
    if hasattr(args, key):
      setattr(args, key, value)

torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))



w_vectorizer = WordVectorizer('./glove', 'our_vab')

if args.dataname == 'kit' : 
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'  
    args.nb_joints = 21
    
else :
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

# wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
# eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Dataloader ---- #####
import psutil
import os
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1e9

logger.info('Creating training dataloader...')
train_loader = dataset_VQ.DATALoader(args.dataname,
                                        args.batch_size,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t,
                                        split='train')

logger.info('Creating validation dataloader...')
val_loader = dataset_VQ.DATALoader(args.dataname,
                                        args.batch_size,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t,
                                        split='val')

mem_after = process.memory_info().rss / 1e9
logger.info(f'Dataloaders created. Memory usage: {mem_before:.2f} GB -> {mem_after:.2f} GB (+{mem_after - mem_before:.2f} GB)')

##### ---- Network ---- #####
# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available!")

logger.info(f'CUDA Device: {torch.cuda.get_device_name(0)}')
logger.info(f'CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

logger.info('Building model...')
import time
start_time = time.time()
net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate,
                       args.vq_act,
                       args.vq_norm)
logger.info(f'Model built in {time.time() - start_time:.2f} seconds')

if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)

logger.info('Moving model to CUDA...')
start_time = time.time()
net = net.cuda()
net.train()
logger.info(f'Model moved to CUDA in {time.time() - start_time:.2f} seconds')

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
  

Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)

#Wandb logging
now = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb.init(project="T2M-GPT-VQ_VAE", 
           config=args,
           name=args.exp_name if train_mode else f"VQVAE_{args.dataname}_{now}",
           entity="malulekevon")

# Update args with wandb sweep parameters (if running a sweep)
for key in wandb.config.keys():
    if hasattr(args, key):
        setattr(args, key, wandb.config[key])

##### ------ warm-up ------- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

for nb_iter in tqdm(range(1, args.warm_up_iter), desc="Warmup", leave=False):
    
  for gt_motion in train_loader:

    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
    
    gt_motion = gt_motion.cuda().float() # (bs, 64, dim)

    pred_motion, loss_commit, perplexity = net(gt_motion)
    loss_motion = Loss(pred_motion, gt_motion)
    loss_vel = Loss.forward_vel(pred_motion, gt_motion)
    
    loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        
        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

##### ---- Training ---- #####
avg_recons, avg_perplexity, avg_commit, avg_total_loss = 0., 0., 0., 0.
# best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper)

# Track best validation loss for checkpoint saving
best_val_loss = float('inf')

# for nb_iter in tqdm(range(1, args.total_iter + 1), desc="Training", leave=False):
nb_iter = 0
while nb_iter < args.total_iter + 1:
    with tqdm(total=args.total_iter, desc="Training", leave=False) as pbar:
      for gt_motion in train_loader :
        
        nb_iter += 1
        pbar.update(1)
        gt_motion = gt_motion.cuda().float() # bs, nb_joints, joints_dim, seq_len
        
        pred_motion, loss_commit, perplexity = net(gt_motion)
        loss_motion = Loss(pred_motion, gt_motion)
        loss_vel = Loss.forward_vel(pred_motion, gt_motion)
        
        loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()
        avg_total_loss += loss.item()
        
        if nb_iter % args.print_iter ==  0 :
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter
            avg_total_loss /= args.print_iter
            
            # Save training metrics before resetting
            train_recons = avg_recons
            train_ppl = avg_perplexity
            train_commit = avg_commit
            train_total_loss = avg_total_loss
            
            writer.add_scalar('./Train/L1', train_recons, nb_iter)
            writer.add_scalar('./Train/PPL', train_ppl, nb_iter)
            writer.add_scalar('./Train/Commit', train_commit, nb_iter)
            writer.add_scalar('./Train/TotalLoss', train_total_loss, nb_iter)
            
            logger.info(f"Train. Iter {nb_iter} : \t Total. {train_total_loss:.5f} \t Commit. {train_commit:.5f} \t PPL. {train_ppl:.2f} \t Recons.  {train_recons:.5f}")
            
            wandb.log({
                "Train/Recons": train_recons,
                "Train/PPL": train_ppl,
                "Train/Commit": train_commit,
                "Train/TotalLoss": train_total_loss,
                "Train/Iter": nb_iter, 
            }, step=nb_iter)
            
            avg_recons, avg_perplexity, avg_commit, avg_total_loss = 0., 0., 0., 0.

            net.eval()
            #Disable gradient computation and reduce memory
            avg_recons_val, avg_perplexity_val, avg_commit_val, avg_total_loss_val = 0., 0., 0., 0.

            with torch.no_grad():
              for gt_motion_val in val_loader:
                    gt_motion_val = gt_motion_val.cuda().float() # bs, nb_joints, joints_dim, seq_len
                    pred_motion_val, loss_commit_val, perplexity_val= net(gt_motion_val)
                    loss_vel_val = Loss.forward_vel(pred_motion_val, gt_motion_val)
                    loss_motion_val = Loss(pred_motion_val, gt_motion_val)

                    loss_val = loss_motion_val + args.commit * loss_commit_val + args.loss_vel * loss_vel_val

                    avg_recons_val += loss_motion_val.item()
                    avg_perplexity_val += perplexity_val.item()
                    avg_commit_val += loss_commit_val.item()
                    avg_total_loss_val += loss_val.item()

              avg_recons_val /= len(val_loader)
              avg_perplexity_val /= len(val_loader)
              avg_commit_val /= len(val_loader)
              avg_total_loss_val /= len(val_loader)

              writer.add_scalar('./Val/L1_Recon', avg_recons_val, nb_iter)
              writer.add_scalar('./Val/PPL', avg_perplexity_val, nb_iter)
              writer.add_scalar('./Val/Commit', avg_commit_val, nb_iter)
              writer.add_scalar('./Val/TotalLoss', avg_total_loss_val, nb_iter)

              wandb.log({
                  "Val/Recons": avg_recons_val,
                  "Val/PPL": avg_perplexity_val,
                  "Val/Commit": avg_commit_val,
                  "Val/TotalLoss": avg_total_loss_val,
                  "Val/Iter": nb_iter, 
              }, step=nb_iter)


              logger.info(f"Val. Iter {nb_iter} : \t Total. {avg_total_loss_val:.5f} \t Commit. {avg_commit_val:.5f} \t PPL. {avg_perplexity_val:.2f} \t Recons.  {avg_recons_val:.5f} ")
              
              writer.add_scalars('./Total/Training vs. Validation VQLoss',{ 'Training' : train_total_loss, 
              'Validation' : avg_total_loss_val },
                nb_iter)
              
              # Save checkpoint only if validation loss improves
              if avg_total_loss_val < best_val_loss:
                  best_val_loss = avg_total_loss_val
                  
                  model_path = f'{args.out_dir}/models'
                  os.makedirs(model_path, exist_ok=True)
                  
                  # Save the best model
                  best_model_path = f'{model_path}/best_model.pth'
                  torch.save({'net' : net.state_dict(),
                              'model_architectue': net,
                              'optimizer': optimizer.state_dict(),
                              'learning_rate': optimizer.param_groups[0]['lr'],
                              'scheduler': scheduler.state_dict(),
                              'iter': nb_iter,
                              'best_val_loss': best_val_loss}, 
                            best_model_path)
                  
                  logger.info(f'✓ Validation loss improved to {best_val_loss:.5f}! Saved checkpoint: {best_model_path}')
                  
                  wandb.log({
                      "Best/Val_Loss": best_val_loss,
                      "Best/Iter": nb_iter,
                  }, step=nb_iter)
              else:
                  logger.info(f'Validation loss did not improve (best: {best_val_loss:.5f}, current: {avg_total_loss_val:.5f})')
              
              net.train()


wandb.finish()