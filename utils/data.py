import numpy as np
import pandas as pd
import trimesh
from matplotlib import pyplot   
from mpl_toolkits.mplot3d import Axes3D 
import random   
import logging
from numpy import linalg as LA
from sklearn.neighbors import KDTree

logging.getLogger("trimesh").setLevel(logging.ERROR)

def show_pc(X_iso):
    fig = pyplot.figure()
    ax = Axes3D(fig)
    sequence_containing_x_vals = list(X_iso[:, 0:1]) 
    sequence_containing_y_vals = list(X_iso[:, 1:2]) 
    sequence_containing_z_vals = list(X_iso[:, 2:3])
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals) 
    pyplot.show()



def normalize(points):
	point_number = points.shape[0]
	feature_number = points.shape[1]

	expectation = np.mean(points, axis=0)

	normalized_points = points-expectation
	l2_norm = LA.norm(normalized_points,axis=1)
	max_distance = max(l2_norm)

	normalized_points = normalized_points/max_distance

	return normalized_points




def load_obj(filepath):
    mesh = trimesh.load(filepath)
    # return np.asarray(mesh.vertices)
    return normalize(mesh.sample(2048)).astype('float32')




def basis_point_set_random(radius, sample_points_number):
	sphere_points = []

	while sample_points_number != 0:

		point = np.array([random.uniform(-radius,radius),random.uniform(-radius,radius),random.uniform(-radius,radius)])

		if np.sum(np.power(point,2)) < pow(radius,2):
			sphere_points.append(point)
			sample_points_number -= 1
	
	sphere_points = np.array(sphere_points)

	return sphere_points


def feature_calculation(normalized_points, sphere_points):
	model_tree = KDTree(normalized_points)
	return model_tree.query(sphere_points)[0]
	

def load_hks(filepath_hks):
	data = pd.read_csv(filepath_hks, header=None)
	matrix = data.to_numpy()
	features = np.linalg.norm(matrix, axis=1)
	features.sort()
	return features

	


def load_HKS_features(filepath, filepath_hks):

	mesh = trimesh.load(filepath)
	points = mesh.sample(1024)
	normalized_points = normalize(points)
	features = np.zeros(shape=(1024), dtype='float32')
	try:
		features = load_hks(filepath_hks)
	except:
		print("error")
	return features.astype('float32'), normalized_points.astype('float32')


def load_obj_features(filepath, basis_pset):
    mesh = trimesh.load(filepath)
    # return np.asarray(mesh.vertices)
    points = mesh.sample(2048)
    normalized_points = normalize(points)
    return feature_calculation(normalized_points, basis_pset).astype('float32'), normalized_points.astype('float32')


# def load_obj_features3(filepath, basis_pset):
#     mesh = trimesh.load(filepath)
#     # return np.asarray(mesh.vertices)
#     points = mesh.sample(2048)
#     normalized_points = normalize(points)
# 	# features = feature_calculation(normalized_points, basis_pset).squezze()
#     return feat.astype('float32'), normalized_points.astype('float32')


def load_obj_features2(filepath, basis_pset):
	mesh = trimesh.load(filepath)
	points = mesh.sample(1024)
	normalized_points= normalize(points)
	temp = np.squeeze(feature_calculation(normalized_points, basis_pset))
	return temp.astype('float32'), normalized_points.astype('float32')
	