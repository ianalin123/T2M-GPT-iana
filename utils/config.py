import os

GMM_MODEL_DIR = "./visualize/joints2smpl/smpl_models/"

SMPL_DATA_PATH = "/home/malulekevon/essentials/body_models/smpl"
SMPL_KINTREE_PATH = os.path.join(GMM_MODEL_DIR, "kintree_table.pkl")
SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "SMPL_NEUTRAL.pkl")
JOINT_REGRESSOR_TRAIN_EXTRA = "/home/malulekevon/essentials/body_model_utils/joint_regressors/J_regressor_h36m.npy"
ROT_CONVENTION_TO_ROT_NUMBER = {
    'legacy': 23,
    'no_hands': 21,
    'full_hands': 51,
    'mitten_hands': 33,
}

GENDERS = ['neutral', 'male', 'female']
NUM_BETAS = 10