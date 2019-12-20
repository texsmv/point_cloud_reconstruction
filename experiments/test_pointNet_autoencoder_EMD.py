import logging
import random
from datetime import datetime
from importlib import import_module
from itertools import chain
from os.path import join, exists
from matplotlib import pyplot

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

from datasets.Vessels import VesselDataset, VesselDataset2
# from losses.earth_mover_distance import EMD
from losses.champfer_loss import ChamferLoss
from utils.pcutil import plot_3d_point_cloud, save_point_cloud,  show_pc, F1, completitud, precision
from utils.util import save_weights, load_latest_epoch, find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging
from models.autoencoder_pointNet import Generator, Encoder




results_dir = "results/"
experiment = "pointNet_ModelNet10_EMD"
save_frequency = 1
max_epochs = 40
batch_size = 30



# def main(config):
# global results_dir, max_epochs, batch_size


#setting directory to save results and weights
results_dir = join(results_dir , experiment)
results_dir = prepare_results_dir(results_dir, b_clean=False)
weights_path = join(results_dir, 'weights')

#finding last saved epoch if exists in resutls directory
starting_epoch = find_latest_epoch(results_dir) + 1

#setting device for pytorch usage
device = cuda_setup(False, 0)

#load vessels dataset
# dataset = VesselDataset2(root_dir="/media/D/Datasets/Tesis/Models with holes")
dataset = VesselDataset2(root_dir="/home/texs/Documents/Repositorios/point_cloud_reconstruction/data/ModelNet10")
points_dataloader = DataLoader(dataset,batch_size= batch_size, shuffle = True, num_workers = 8, drop_last=True, pin_memory=True)

#loading models and weights
G = Generator().to(device)
E = Encoder().to(device)


EG_optim = torch.optim.Adam(chain(E.parameters(), G.parameters()), lr= 0.0005, weight_decay= 0, betas= [0.9, 0.999],amsgrad= False)



#loading weights if they exists in results directory
load_latest_epoch(E, G, EG_optim, weights_path ,starting_epoch)




counter = 0.0
score_completitud = 0.0
score_precision = 0.0
score_F1 = 0.0
for i, point_data in enumerate(points_dataloader, 0):

    X, _ = point_data
    X = X.to(device)
    if X.size(-1) == 3:
        X.transpose_(X.dim() - 2, X.dim() - 1)
    with torch.no_grad():
        X_rec = G(E(X))

    if X_rec.size(-1) != 3:
        X_rec.transpose_(X_rec.dim() - 1, X_rec.dim() - 2)
        X.transpose_(X.dim() - 1, X.dim() - 2)

    for i in range (batch_size):
        f1 = F1(X_rec[i], X[i], 0.09)
        c = completitud(X_rec[i], X[i], 0.09)
        p = precision(X_rec[i], X[i], 0.09)

        print("f1: ", f1, " comp: ",c, "prec: ", p)
        
        counter += 1.0
        score_F1 += f1
        score_completitud += c
        score_precision += p

        # show_pc(X_rec[i])
        # pyplot.show()
        # show_pc(X[i])
        # show_pc(X_c[0])
# X, Y = iter(points_dataloader).next()
# print(Y.shape)


# X_c = X.clone()

# # for i in range(X.size()[0]):
# #     for j in range(250):
# #         X[i][j] = 0

# X.transpose_(X.dim() - 2, X.dim() - 1)



# with torch.no_grad():
#     print(E(X).shape)
#     X_rec = G(E(X))


# if X_rec.size(-1) != 3:
#     X_rec.transpose_(X_rec.dim() - 2, X_rec.dim() - 1)
#     # X.transpose_(X.dim() - 1, X.dim() - 2)
# print("Rec shape: ", X_rec.shape)

# for i in range (batch_size):
#     print(F1(X_rec[i], Y[i], 0.05))
#     print(completitud(X_rec[i], Y[i], 0.1))
#     print(precision(X_rec[i], Y[i], 0.1))
#     show_pc(X_rec[i])
#     show_pc(Y[i])

#     # show_pc(X_c[0])
#     print(X_rec.shape)
#     print(X_c.shape)
#     pyplot.show()





#     # training
#     # for epoch in range(starting_epoch, max_epochs ):
#     #     start_epoch_time = datetime.now()

#     #     total_loss = 0.0
#     #     for i, point_data in enumerate(points_dataloader, 0):
#     #         X, _ = point_data
#     #         X = X.to(device)

#     #         # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
#     #         if X.size(-1) == 3:
#     #             X.transpose_(X.dim() - 2, X.dim() - 1)



#     #     with torch.no_grad():
#     #         X_rec = G(E(X))


#     #     if X_rec.size(-1) != 3:
#     #             X_rec.transpose_(X_rec.dim() - 1, X_rec.dim() - 2)
#     #             X.transpose_(X.dim() - 1, X.dim() - 2)

        
#     #     X_rec = X_rec.cpu().numpy()
#     #     X = X.cpu().numpy()

#     #     show_pc(X_rec[0])
#     #     show_pc(X[0])
#     #     pyplot.show()

        
        

# # if __name__ == '__main__':
# #     main(1)
        # pyplot.show()

    if(i >=1):
        break


print("mean F1: ", score_F1/counter, " mean Compl: ", score_completitud/counter, " mean Prec: ", score_precision/counter)












# X = iter(points_dataloader).next()[0]

# X_c = X.clone()

# # for i in range(X.size()[0]):
# #     for j in range(1500):
# #         X[i][j] = 0

# X.transpose_(X.dim() - 2, X.dim() - 1)


# print(X.shape)

# with torch.no_grad():
#     print(E(X).shape)
#     X_rec = G(E(X))


# if X_rec.size(-1) != 3:
#     X_rec.transpose_(X_rec.dim() - 1, X_rec.dim() - 2)
#     X.transpose_(X.dim() - 1, X.dim() - 2)


# for i in range (batch_size):
#     print(F1(X_rec[i], X[i], 0.06))
#     print(completitud(X_rec[i], X[i], 0.06))
#     print(precision(X_rec[i], X[i], 0.06))
#     show_pc(X_rec[i])
#     show_pc(X[i])
#     # show_pc(X_c[0])
#     pyplot.show()

#     # training
#     # for epoch in range(starting_epoch, max_epochs ):
#     #     start_epoch_time = datetime.now()

#     #     total_loss = 0.0
#     #     for i, point_data in enumerate(points_dataloader, 0):
#     #         X, _ = point_data
#     #         X = X.to(device)

#     #         # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
#     #         if X.size(-1) == 3:
#     #             X.transpose_(X.dim() - 2, X.dim() - 1)



#     #     with torch.no_grad():
#     #         X_rec = G(E(X))


#     #     if X_rec.size(-1) != 3:
#     #             X_rec.transpose_(X_rec.dim() - 1, X_rec.dim() - 2)
#     #             X.transpose_(X.dim() - 1, X.dim() - 2)

        
#     #     X_rec = X_rec.cpu().numpy()
#     #     X = X.cpu().numpy()

#     #     show_pc(X_rec[0])
#     #     show_pc(X[0])
#     #     pyplot.show()

        
        

# # if __name__ == '__main__':
# #     main(1)