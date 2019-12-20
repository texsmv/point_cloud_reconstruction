
import numpy as np
import tensorflow as tf
import pickle
import os

from os.path import exists, join
from os import listdir, makedirs, remove
from utils.data import load_obj, load_obj_features2, basis_point_set_random

# root_dir = "/home/texs/Documents/Repositorios/point_cloud_reconstruction/data/SimplifiedManifolds"
root_dir = "/home/texs/Documents/Repositorios/point_cloud_reconstruction/data/ModelNet10"
pc_names = np.array(listdir(root_dir))
pc_names.sort()




def get_number_pc():
    return len(pc_names)

def generator(n_pc):
    idx = 0
    while(idx < n_pc):
    # for idx in range(len(pc_names)):
        if idx == len(pc_names):
            idx = 0
        pc_filename = pc_names[idx]
        pc_filepath = join(root_dir, pc_filename)
        points = load_obj(pc_filepath)
        idx += 1
        yield points