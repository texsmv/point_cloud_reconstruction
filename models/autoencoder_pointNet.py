import torch.nn as nn

z_size = 512
g_use_bias = True
e_use_bias = True
slope = 0.2

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.z_size = z_size
        self.use_bias = g_use_bias

        self.model = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=64,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=128,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=512,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=1024,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=2048 * 3,
                      bias=self.use_bias),
        )

    def forward(self, z):
        output = self.model(z.squeeze())
        output = output.view(-1, 3, 2048)
        return output



class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.z_size = z_size
        self.use_bias = e_use_bias
        self.relu_slope = slope

        self.pc_encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=self.z_size, kernel_size=1,
                      bias=self.use_bias),
        )

        # self.pc_encoder_fc = nn.Sequential(
        #     nn.Linear(512, 256, bias=True),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Linear(256, self.z_size, bias=True)
        # )

    def forward(self, x):
        output = self.pc_encoder_conv(x)
        output = output.max(dim=2)[0]
        # output = self.pc_encoder_fc(output)
        return output
