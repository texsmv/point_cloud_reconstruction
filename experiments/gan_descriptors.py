
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.autograd import Variable
from datasets.Vessels import VesselDataset, VesselDataset2, VesselDataset_Pset
from torch.utils.data import DataLoader
from datetime import datetime
from utils.util import save_weights, load_latest_epoch, find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging
from utils.pcutil import plot_3d_point_cloud, save_point_cloud,  show_pc, F1, completitud, precision
from models.autoencoder_basis_pset import Generator, Encoder
from itertools import chain
from os.path import join, exists
from matplotlib import pyplot

device = cuda_setup(True, 0)

results_dir = "results/"
experiment = "basisPnet_ModelNet10"

results_dir = join(results_dir , experiment)
results_dir = prepare_results_dir(results_dir, b_clean=False)
weights_path = join(results_dir, 'weights')



random.seed(2019)
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)

starting_epoch = find_latest_epoch(results_dir) + 1

G = Generator().to(device)
E = Encoder().to(device)


EG_optim = torch.optim.Adam(chain(E.parameters(), G.parameters()), lr= 0.0005, weight_decay= 0, betas= [0.9, 0.999],amsgrad= False)

load_latest_epoch(E, G, EG_optim, weights_path ,starting_epoch)



max_epochs = 40
batch_size = 30

def weights_init(m):
    classname = m.__class__.__name__
    if classname in ('Conv1d', 'Linear'):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def get_dataset():
    return torch.Tensor(np.random.normal(4,1.25,(batch_size,1000)))


def make_noise():
    return torch.Tensor(np.random.uniform(0,1,(batch_size,50)))


class generator(nn.Module):
    
    def __init__(self,inp,out):
        super(generator,self).__init__()
        self.net = nn.Sequential(nn.Linear(inp,300),
                                nn.BatchNorm1d(300),
                                 nn.ReLU(inplace=True),
                                nn.Linear(300,500),
                                nn.BatchNorm1d(500),
                                 nn.ReLU(inplace=True),
                                nn.Linear(500,out),
                                nn.BatchNorm1d(out),
                                nn.ReLU(inplace=True)
                                # nn.Tanh()
                               )
        
    def forward(self,x):
        x = self.net(x)
        return x

    
class discriminator(nn.Module):
    
    def __init__(self,inp,out):
        super(discriminator,self).__init__()
        self.net = nn.Sequential(nn.Linear(inp,500),
                                # nn.BatchNorm1d(500),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(500,300),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(300,out),
                                 nn.Sigmoid()
                                )
        
    def forward(self,x):
        x = self.net(x)
        return x


def stats(array):
    array = array.detach().numpy()
    return [np.mean(array),np.std(array)]


results_dirGan = "results/"
experimentGan = "gan_ModelNet10"

results_dirGan = join(results_dirGan , experimentGan)
results_dirGan = prepare_results_dir(results_dirGan, b_clean=False)
weights_pathGan = join(results_dirGan, 'weights')

# starting_epochGan = find_latest_epoch(results_dirGan) + 1

device = cuda_setup(True, 0)

dataset = VesselDataset_Pset(root_dir="/home/texs/Documents/Repositorios/point_cloud_reconstruction/data/ModelNet10")
# dataset = VesselDataset_Pset()
points_dataloader = DataLoader(dataset,batch_size= batch_size, shuffle = True, num_workers = 8, drop_last=True, pin_memory=True)


log = logging.getLogger(__name__)
# X = iter(points_dataloader).next()[0]
# X = X.squeeze()
# print(X.shape)

# gen = generator(50,1000)
# discrim = discriminator(1000,1)

gen = generator(50,1000).to(device)
discrim = discriminator(1000,1).to(device)

# gen.apply(weights_init)
# discrim.apply(weights_init)


epochs = 500

d_step = 10
g_step = 8

criteriond1 = nn.BCELoss().to(device)
optimizerd1 = optim.SGD(discrim.parameters(), lr=0.001, momentum=0.9)

criteriond2 = nn.BCELoss().to(device)
optimizerd2 = optim.SGD(gen.parameters(), lr=0.001, momentum=0.9)

printing_steps = 20



for epoch in range(epochs):
    print("Epoch: ", epoch)
    # gen.train()
    # discrim.train()

    g_total_loss = 0.0
    d_total_loss = 0.0
    for i, point_data in enumerate(points_dataloader, 0):
        X, F = point_data
        
        # print("->", X.shape)
        # X = X.squeeze()
        X = X.reshape([30,1000])
        X_c = X.clone()
        # print("-->", X.shape)   
        # print(i)
        # if epoch%printing_steps==0:
        #     print("Epoch:", epoch)
        Y = get_dataset()
        # training discriminator
        X = X.to(device)
        Y = Y.to(device)












# -------------------------------------------------------------------------------
        # for d_i in range(d_step):
        discrim.zero_grad()
        
        #real
        # data_d_real = Variable(get_dataset())
        data_d_real = Variable(X)
        # data_d_real = X
        # print(data_d_real.shape)
        data_d_real_pred = discrim(data_d_real)
        data_d_real_loss = criteriond1(data_d_real_pred.to(device),Variable(torch.ones(batch_size,1)).to(device))
        data_d_real_loss.backward()

        d_total_loss += data_d_real_loss.item()
        
        #fake
        data_d_noise = Variable(make_noise())
        data_d_gen_out = gen(data_d_noise.to(device)).detach()
        data_fake_dicrim_out = discrim(data_d_gen_out).to(device)
        data_fake_d_loss = criteriond1(data_fake_dicrim_out,Variable(torch.zeros(batch_size,1)).to(device))
        data_fake_d_loss.backward()

        d_total_loss += data_fake_d_loss.item()
        

        optimizerd1.step()

            



        # ----------------------------------------------------------------------------------------
        # for g_i in range(g_step):
        gen.zero_grad()
        
        data_noise_gen = Variable(make_noise())
        data_g_gen_out = gen(data_noise_gen.to(device))
        data_g_dis_out = discrim(data_g_gen_out).to(device)
        data_g_loss = criteriond2(data_g_dis_out,Variable(torch.ones(batch_size,1)).to(device))
        data_g_loss.backward()

        g_total_loss += data_g_loss.item()
        
        optimizerd2.step()
            
            # if epoch%printing_steps==0:
            #     print(stats(data_g_gen_out))
    
    print(
        f'[{epoch}/{epochs}] '
        f'D Loss: {d_total_loss / i:.4f} '
        f'G Loss: {g_total_loss / i:.4f} '
    )

    if(epoch % 5 == 0):

        if F.size(-1) == 3:
            F.transpose_(F.dim() - 2, F.dim() - 1)
        with torch.no_grad():
            X_rec = G(E(data_g_gen_out.to(device)))

        if X_rec.size(-1) != 3:
            X_rec.transpose_(X_rec.dim() - 1, X_rec.dim() - 2)
        # print(X_rec.shape)
        # print(X_c.shape)
        data = data_g_gen_out.cpu()
        data_r = X.cpu()
        # print(data[0])
        # print(data_r[0])
        X_rec = X_rec.cpu()
        show_pc(X_rec[0])
        show_pc(X_rec[1])
        show_pc(X_rec[2])
        show_pc(X_rec[3])
        # show_pc(X[0])
        # show_pc(F[0])
        pyplot.show()

    # print(stats(data_g_gen_out.cpu()))
    

    # log.debug(
    #     f'[{epoch}/{epochs}] '
        
    #     f'Time: {datetime.now() - start_epoch_time}'
    # )
    
    