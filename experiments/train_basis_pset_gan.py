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

from datasets.Vessels import VesselDataset, VesselDataset2, VesselDataset3
from losses.champfer_loss import ChamferLoss
from utils.pcutil import plot_3d_point_cloud, save_point_cloud
from utils.util import save_weights, load_latest_epoch, find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging
from models.autoencoder_basis_pset import Generator, Encoder


results_dir = "results/"
experiment = "basisPnetGan"
save_frequency = 1
max_epochs = 40
batch_size = 30
n_points = 1024

def weights_init(m):
    classname = m.__class__.__name__
    if classname in ('Conv1d', 'Linear'):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)



def main(config):
    global results_dir, max_epochs, batch_size

    # seeds
    random.seed(2019)
    torch.manual_seed(2019)
    torch.cuda.manual_seed_all(2019)


    #setting directory to save results and weights
    results_dir = join(results_dir , experiment)
    results_dir = prepare_results_dir(results_dir, b_clean=False)
    weights_path = join(results_dir, 'weights')

    # find latest epoch
    # starting_epoch = find_latest_epoch(results_dir) + 1


    device = cuda_setup(True, 0)

    log = logging.getLogger(__name__)

    dataset = VesselDataset3()


    points_dataloader = DataLoader(dataset,batch_size= batch_size, shuffle = True, num_workers = 8, drop_last=True, pin_memory=True)
    noise = tf.placeholder(tf.float32, [None, n_points, 3])
    

    G = Generator().to(device)
    E = Encoder().to(device)

    G.apply(weights_init)
    E.apply(weights_init)


    reconstruction_loss = ChamferLoss().to(device)


    EG_optim = torch.optim.Adam(chain(E.parameters(), G.parameters()), lr= 0.0005, weight_decay= 0, betas= [0.9, 0.999],amsgrad= False)

    # load_latest_epoch(E, G, EG_optim, weights_path ,starting_epoch)



    for epoch in range(0, max_epochs ):
        start_epoch_time = datetime.now()

        G.train()
        E.train()

        total_loss = 0.0
        for i, point_data in enumerate(points_dataloader, 0):
            F, X = point_data
            X = X.to(device)
            F = F.to(device)
            # print(X.shape)

            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if X.size(-1) == 3:
                X.transpose_(X.dim() - 2, X.dim() - 1)

            # print("features shape: ", F.shape)
            # print("points shape: ", X.shape)
            
            X_rec = G(E(F))
            # print("reconstructed points shape: ", X_rec.shape)

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
            X_rec = G(E(F)).data.cpu().numpy()

        X_cpu = X.cpu().numpy()
        print(X_rec.min(axis=0), X_rec.max(axis=0))
        print(X_cpu.min(axis=0), X_cpu.max(axis=0))

        
        # save_point_cloud(X_cpu, X_rec, epoch, n_fig=5, results_dir=results_dir)

        # if epoch % save_frequency == 0:
        #     save_weights(E, G, EG_optim, weights_path, epoch)
        

if __name__ == '__main__':
    main(1)