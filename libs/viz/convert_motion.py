import numpy as np
from visualize.simplify_loc2rot import joints2smpl
from model.rotation2xyz import Rotation2xyz
import torch
#Convert Pose_xyz to SMPL


def convert_motion(motion_path):
  
    motions = np.load(motion_path, allow_pickle=True)[None][0]
    frames, njoints, nfeats = motions.shape
    MINS = motions.min(axis=0).min(axis=0)
    MAXS = motions.max(axis=0).max(axis=0)

    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset
    trajec = motions[:, 0, [0, 2]]
    
    j2s = joints2smpl(num_frames=frames, device_id=0, cuda=True)
    rot2xyz = Rotation2xyz(device=torch.device("cuda:0"))
    faces = rot2xyz.smpl_model.faces
    _, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3]
    pose = opt_dict['pose'][:,3:].reshape(-1, -1, 3)
    orient = opt_dict['pose'][:, :3].reshape(-1, 3)
    betas = opt_dict['betas']
    transl = opt_dict['cam']
    
    return orient, betas, transl