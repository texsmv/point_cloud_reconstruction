import torch.nn as nn

feature_size = 1000
pc_size = 2048

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.Tanh())
            
        
    def forward(self, x):
        output = self.encoder(x.squeeze())
        return output



class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, pc_size * 3),
            # nn.ReLU(True),
            # nn.Linear(pc_size * 3, pc_size * 3),
            nn.Tanh())
    def forward(self, x):
        output = self.decoder(x.squeeze())
        output = output.view(-1, 3, 2048)
        return output