
import numpy as np
import tensorflow as tf
import pickle
import os

from os.path import exists, join
from os import listdir, makedirs, remove
from utils.data import load_obj, load_obj_features2, basis_point_set_random


N_BASIS = 1024

root_dir = "/home/texs/Documents/Repositorios/point_cloud_reconstruction/data/SimplifiedManifolds"
pc_names = np.array(listdir(root_dir))
pc_names.sort()

basis_pset = []

if os.path.exists("basis.pkl"):
    print("-----------------------------LOADING BASIS POINT-------------------------------")
    basis_pset = pickle.load( open( "basis.pkl", "rb" ) )
else:
    basis_pset = basis_point_set_random(1.0, N_BASIS)
    pickle.dump( basis_pset, open( "basis.pkl", "wb" ) )


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
        features, points = load_obj_features2(pc_filepath, basis_pset)
        idx += 1
        yield features, points