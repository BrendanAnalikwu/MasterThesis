import torch


class SurrogateNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer1 = torch.nn.Conv2d(4, 4, 2, 1, 0)
        self.layer2 = torch.nn.Sequential(
            # 128
            torch.nn.Conv2d(6, 8, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 126
            torch.nn.Conv2d(8, 8, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 124
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            # 62
            torch.nn.Conv2d(8, 16, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 60
            torch.nn.Conv2d(16, 16, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 58
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            # 29
            torch.nn.Conv2d(16, 32, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 27
            torch.nn.Conv2d(32, 32, 3, 2, 0),
            torch.nn.LeakyReLU(.1),
            # 13
            torch.nn.Conv2d(32, 32, 3, 1, 1),
            torch.nn.LeakyReLU(.1),
            # 13
            torch.nn.Conv2d(32, 32, 3, 2, 0),
            torch.nn.LeakyReLU(.1),
            # 6
            torch.nn.Flatten(1, -1),
            # 2304
            torch.nn.Linear(1152, 512),
            torch.nn.LeakyReLU(.1),
            torch.nn.Linear(512, 512),
            torch.nn.LeakyReLU(.1),
            torch.nn.Linear(512, 1152),
            torch.nn.Unflatten(-1, (32, 6, 6)),
            # 6
            torch.nn.ConvTranspose2d(32, 32, 3, 2, 0),
            torch.nn.ReLU(),
            # 13
            torch.nn.Conv2d(32, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, 3, 2, 0),
            torch.nn.ReLU(),
            # 27
            torch.nn.ConvTranspose2d(32, 16, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 29
            torch.nn.ConvTranspose2d(16, 16, 2, 2, 0),
            torch.nn.LeakyReLU(.1)
            # 58
        )
        self.layer5 = torch.nn.Sequential(
            # 58
            torch.nn.ConvTranspose2d(32, 16, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 60
            torch.nn.ConvTranspose2d(16, 16, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 62
            torch.nn.ConvTranspose2d(16, 8, 2, 2, 0),
            torch.nn.LeakyReLU(.1),
            # 124
        )
        self.layer6 = torch.nn.Sequential(
            # 124
            torch.nn.ConvTranspose2d(16, 8, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 126
            torch.nn.ConvTranspose2d(8, 8, 3, 1, 0),
            torch.nn.LeakyReLU(.1)
            # 128
        )
        self.layer7 = torch.nn.Sequential(torch.nn.ConvTranspose2d(8, 4, 2, 1, 0),
                                          torch.nn.LeakyReLU(.1),
                                          torch.nn.ConvTranspose2d(4, 2, 3, 1, 1),
                                          torch.nn.Tanh())

    def forward(self, v, H, A, v_a):
        x = self.layer1(torch.cat((v, v_a), 1))
        x = torch.cat((x, H[:, None, ...], A[:, None, ...]), 1)
        x1 = self.layer2(x)  # 124
        x2 = self.layer3(x1)  # 58
        x3 = self.layer4(x2)
        x4 = self.layer5(torch.cat((x3, x2), 1))
        x5 = self.layer6(torch.cat((x4, x1), 1))
        dv = self.layer7(x5)

        return v + .1 * dv  # , torch.nn.functional.relu(x5[:, 2, :, :]), torch.nn.functional.sigmoid(x5[:, 3, :, :])
