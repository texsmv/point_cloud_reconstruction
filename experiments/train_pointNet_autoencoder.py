import logging
import random
from datetime import datetime
from importlib import import_module
from itertools import chain
from os.path import join, exists

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

from datasets.Vessels import VesselDataset, VesselDataset2
from losses.champfer_loss import ChamferLoss
from losses.earth_mover_distance import EMD
from utils.pcutil import plot_3d_point_cloud, save_point_cloud
from utils.util import save_weights, load_latest_epoch, find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging
from models.autoencoder_pointNet import Generator, Encoder


results_dir = "results/"
experiment = "pointNet_ModelNet10"
save_frequency = 1
max_epochs = 40
batch_size = 30

def weights_init(m):
    classname = m.__class__.__name__
    if classname in ('Conv1d', 'Linear'):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)



def main(config):
    global results_dir, max_epochs, batch_size

    # setting seeds
    random.seed(2019)
    torch.manual_seed(2019)
    torch.cuda.manual_seed_all(2019)

    #setting directory to save results and weights
    results_dir = join(results_dir , experiment)
    results_dir = prepare_results_dir(results_dir, b_clean=False)
    weights_path = join(results_dir, 'weights')

    #finding last saved epoch if exists in resutls directory
    starting_epoch = find_latest_epoch(results_dir) + 1

    #setting device for pytorch usage
    device = cuda_setup(True, 0)

    #use to log useful information
    log = logging.getLogger(__name__)

    #load vessels dataset
    dataset = VesselDataset2(root_dir="/home/texs/Documents/Repositorios/point_cloud_reconstruction/data/ModelNet10")
    points_dataloader = DataLoader(dataset,batch_size= batch_size, shuffle = True, num_workers = 8, drop_last=True, pin_memory=True)


    #loading models and weights
    G = Generator().to(device)
    E = Encoder().to(device)
    G.apply(weights_init)
    E.apply(weights_init)

    # setting reconstruction loss
    reconstruction_loss = ChamferLoss().to(device)
    # reconstruction_loss = EMD().to(device)
    # reconstruction_loss = EMD().to(device)


    #optimization in models parameters
    EG_optim = torch.optim.Adam(chain(E.parameters(), G.parameters()), lr= 0.0005, weight_decay= 0, betas= [0.9, 0.999],amsgrad= False)



    #loading weights if they exists in results directory
    load_latest_epoch(E, G, EG_optim, weights_path ,starting_epoch)


    
    # training
    for epoch in range(starting_epoch, max_epochs ):
        start_epoch_time = datetime.now()

        G.train()
        E.train()

        total_loss = 0.0
        for i, point_data in enumerate(points_dataloader, 0):
            X, _ = point_data
            X = X.to(device)

            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if X.size(-1) == 3:
                X.transpose_(X.dim() - 2, X.dim() - 1)

            X_rec = G(E(X))

            loss = torch.mean(0.05 * reconstruction_loss(X.permute(0, 2, 1) + 0.5, X_rec.permute(0, 2, 1) + 0.5))

            EG_optim.zero_grad()
            E.zero_grad()
            G.zero_grad()

            loss.backward()
            total_loss += loss.item()

            EG_optim.step()


            print(f'[{epoch}: ({i})] '
                      f'Loss: {loss.item():.4f} '
                      f'Time: {datetime.now() - start_epoch_time}')
        log.debug(
            f'[{epoch}/{max_epochs}] '
            f'Loss: {total_loss / i:.4f} '
            f'Time: {datetime.now() - start_epoch_time}'
        )
        
        G.eval()
        E.eval()

        with torch.no_grad():
            X_rec = G(E(X)).data.cpu().numpy()

        X_cpu = X.cpu().numpy()

        
        save_point_cloud(X_cpu, X_rec, epoch, n_fig=5, results_dir=results_dir)

        if epoch % save_frequency == 0:
            save_weights(E, G, EG_optim, weights_path, epoch)
        

if __name__ == '__main__':
    main(1)