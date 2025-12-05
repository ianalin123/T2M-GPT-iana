import os
import numpy as np
import torch

import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric

import models.vqvae as vqvae
import options.option_vq as option_vq
from dataset import dataset_TM_eval


def decode_motion(zq, args, val_loader):
    net = vqvae.HumanVQVAE(args,
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
    net.eval()
    pred_motion = net.forward_decoder(zq)  #Decode the motion from the latent code
    pred_denorm = val_loader.dataset.inv_transform(pred_motion.detach().cpu().numpy())  #Inverse transform the motion to the original space
    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), args.nb_joints)  #Recover the motion from the original space
    return pred_xyz

def save_results(zq, args, name, out_dir, val_loader):
    pose_xyz = decode_motion(zq, args, val_loader)
    np.save(os.path.join(out_dir, name+'_linear_probe.npy'), pose_xyz.detach().cpu().numpy())
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--zq_path', type=str, default='results/zq.npy')
    parser.add_argument('--out_dir', type=str, default='results')
    configs = parser.parse_args()
    
    args = option_vq.get_args_parser()
    
    args.nb_joints = 21 if args.dataname == 'kit' else 22
    
    from utils.word_vectorizer import WordVectorizer 
    
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer, unit_length=2**args.down_t)
 
    zq = np.load(args.zq_path, allow_pickle=True)
    
    name = args.zq_path.split('/')[-1].split('.')[0]

    save_results(zq, args, name, args.out_dir, val_loader)