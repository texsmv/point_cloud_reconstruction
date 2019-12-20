from os import listdir, makedirs, remove
from os.path import exists, join
import os
import pickle

import pandas as pd
import numpy as np
import itertools

from torch.utils.data import Dataset
from utils.data import load_obj, load_obj_features, basis_point_set_random, load_HKS_features


class VesselDataset(Dataset):
    def __init__(self, root_dir = "D/3D Models", split = 'train'):
        
        self.root_dir = root_dir
        self.pc_names = []
        self.categories = listdir(root_dir)
        #todo: some of the objets in these categories are corrupted
        self.categories.remove("Amphora")
        self.categories.remove("All Models")
        self.categories.remove("Modern-Glass")
        
        self.split = split
        self.get_names()


    
    def get_names(self):
        self.pc_names = [ [file for file in listdir(join(self.root_dir, category))] for category in self.categories]
        for i in range(len(self.categories)):
            for j in range(len(self.pc_names[i])):
                self.pc_names[i][j] = join(self.categories[i], self.pc_names[i][j])
        if(self.split == 'train'):
            self.pc_names = [ pc[:int(0.85 * len(pc))] for pc in self.pc_names ]
        elif(self.split == 'test'):
            self.pc_names = [ pc[int(0.85 * len(pc)):] for pc in self.pc_names ]

        self.pc_names = list(itertools.chain.from_iterable(self.pc_names))
        self.pc_names = np.array(self.pc_names)
        self.pc_names = self.pc_names.flatten()
        

    
    def __len__(self):
        return len(self.pc_names)
    
    def __getitem__(self, idx):
        pc_filename = self.pc_names[idx]
        pc_filepath = join(self.root_dir, pc_filename)
        sample = load_obj(pc_filepath)
        return sample, 0





class VesselDataset2(Dataset):
    def __init__(self, root_dir = "/media/D/Datasets/Tesis/SimplifiedManifolds"):
    # def __init__(self, root_dir = "/media/data/Datasets/Tesis/3D Models/Abstract"):
        
        self.root_dir = root_dir
        self.pc_names = []
        self.get_names()
    
    def get_names(self):
        self.pc_names = np.array(listdir(self.root_dir))
        self.pc_names.sort()

    
    def __len__(self):
        return len(self.pc_names)
    
    def __getitem__(self, idx):
        pc_filename = self.pc_names[idx]
        pc_filepath = join(self.root_dir, pc_filename)
        sample = load_obj(pc_filepath)
        return sample, 0


class VesselDataset_Pset(Dataset):
    def __init__(self, root_dir = "/home/texs/Documents/Repositorios/point_cloud_reconstruction/data/SimplifiedManifolds"):        
        self.root_dir = root_dir
        self.pc_names = []
        self.get_names()
        self.basis_pset = []
        if os.path.exists("basis_pytorch.pkl"):
            self.basis_pset = pickle.load( open( "basis_pytorch.pkl", "rb" ) )
        else:
            self.basis_pset = basis_point_set_random(1.0, 1000)
            pickle.dump( self.basis_pset, open( "basis_pytorch.pkl", "wb" ) )
    
    def get_names(self):
        self.pc_names = np.array(listdir(self.root_dir))
        self.pc_names.sort()

    
    def __len__(self):
        return len(self.pc_names)
    
    def __getitem__(self, idx):
        pc_filename = self.pc_names[idx]
        pc_filepath = join(self.root_dir, pc_filename)
        features, points = load_obj_features(pc_filepath, self.basis_pset)
        return features, points


class VesselDataset_HKS1(Dataset):
    def __init__(self, root_dir_hks = "/home/texs/Documents/Repositorios/point_cloud_reconstruction/data/150", root_dir_pc = "/home/texs/Documents/Repositorios/point_cloud_reconstruction/data/SimplifiedManifolds"):
        self.root_dir_hks = root_dir_hks
        self.root_dir_pc = root_dir_pc
        self.pc_names = []
        self.get_names()
    
    def get_names(self):
        self.pc_names = np.array(listdir(self.root_dir_pc))
        self.pc_names.sort()
        self.pc_names_hks = np.array(listdir(self.root_dir_hks))
        self.pc_names_hks.sort()

    
    def __len__(self):
        return len(self.pc_names)
    
    def __getitem__(self, idx):
        pc_filename = self.pc_names[idx]
        pc_filepath = join(self.root_dir_pc, pc_filename)
        pc_filename_hks = self.pc_names_hks[idx]
        pc_filepath_hks = join(self.root_dir_hks, pc_filename_hks)
        features, points = load_HKS_features(pc_filepath, pc_filepath_hks)
        return features, points




class VesselDataset4(Dataset):
    def __init__(self, root_dir = "/home/texs/Documents/Repositorios/point_cloud_reconstruction/data/SimplifiedManifolds"):        
        self.root_dir = root_dir
        self.pc_names = []
        self.get_names()
        self.basis_pset = basis_point_set_random(1.0, 1000)
    
    def get_names(self):
        self.pc_names = np.array(listdir(self.root_dir))
        self.pc_names.sort()

    
    def __len__(self):
        return len(self.pc_names)
    
    def __getitem__(self, idx):
        pc_filename = self.pc_names[idx]
        pc_filepath = join(self.root_dir, pc_filename)
        features, points = load_obj_features(pc_filepath, self.basis_pset)
        return features, points



def get_noise(batch_size, n_points):
    return np.random.normal(size=(batch_size, n_points, 3))