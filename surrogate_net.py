from typing import List

import torch


class PatchNet(torch.nn.Module):
    def __init__(self, overlap: int = 1, out_size: int = 3, n_hidden: List[int] = (16, 32, 64), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_layer_v = torch.nn.Conv2d(6, n_hidden[0], 2 * overlap + 1, 1, 0, bias=False)  # -> -2
        if overlap > 0:
            self.input_layer_HA = torch.nn.Conv2d(2, n_hidden[0], 2 * overlap, 1, 0, bias=False)  # -> -1
        else:
            self.input_layer_HA = torch.nn.Sequential(torch.nn.Upsample((out_size, out_size), mode='bilinear'),
                                                      torch.nn.Conv2d(2, n_hidden[0], 1, 1, 0))
        self.input_layer_border = torch.nn.Linear(1, n_hidden[0], bias=True)  # -> 16x1x1
        self.input_activation = torch.nn.Sequential(torch.nn.BatchNorm2d(n_hidden[0]), torch.nn.GELU())

        self.encoder = torch.nn.Sequential(
            # 16x5x5
            torch.nn.Conv2d(n_hidden[0], n_hidden[1], 3, 1, 0),
            torch.nn.BatchNorm2d(n_hidden[1]),
            torch.nn.GELU())
        for i in range(int((out_size - 3) / 2)):
            self.encoder.extend(torch.nn.Sequential(torch.nn.Conv2d(n_hidden[1], n_hidden[1], 3, 1, 0),
                                                    torch.nn.BatchNorm2d(n_hidden[1]),
                                                    torch.nn.GELU()))
        self.encoder.append(
            # 32x2x2
            torch.nn.Conv2d(n_hidden[1], n_hidden[2], 1, 1, 0)
        )
        # self.encoder[-1].bias[-2:].data.fill_(-6.)
        # self.encoder[-1].bias[-4:-2].data.fill_(0.)

        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm2d(n_hidden[2]),
            torch.nn.GELU(),
            # 64x1x1
            torch.nn.ConvTranspose2d(n_hidden[2], n_hidden[0], 3, 1, 0),
            torch.nn.InstanceNorm2d(n_hidden[0]),
            torch.nn.GELU())
        for i in range(int((out_size - 3) / 2)):
            self.decoder.extend(torch.nn.Sequential(torch.nn.Upsample((5 + i * 2, 5 + i * 2), mode='bilinear'),
                                                    torch.nn.Conv2d(n_hidden[0], n_hidden[0], 2, 1, 0),
                                                    torch.nn.BatchNorm2d(n_hidden[0]),
                                                    torch.nn.GELU()))
        self.decoder.extend(torch.nn.Sequential(
            # 16x3x3
            torch.nn.Upsample((out_size + 2, out_size + 2), mode='bilinear'),
            torch.nn.Conv2d(n_hidden[0], 2, 3, 1, 0),
            # torch.nn.Identity(),
            # torch.nn.InstanceNorm2d(2)
        )
            # 2x5x5
        )

        # self.s_activation = torch.nn.Sigmoid()
        # self.m_activation = torch.nn.Identity()

    def forward(self, v, H, A, v_a, v_o, border):
        x = self.input_activation(self.input_layer_v(torch.cat((v, v_a, v_o), 1))
                                  + self.input_layer_HA(torch.cat((H, A), 1))
                                  + self.input_layer_border(border).transpose(1, 3))
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        # x2 = self.decoder(x1[:, :-4])
        # m, s = x1[:, -4:-2], x1[:, -2:]
        return x2  # , self.m_activation(m), self.s_activation(s)


class SurrogateNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer1 = torch.nn.Conv2d(4, 6, 2, 1, 0, bias=False)
        self.layer2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(8),
            torch.nn.GELU(),
            # 256
            torch.nn.Conv2d(8, 8, 4, 2, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(8),
            torch.nn.GELU(),
            # 128
            torch.nn.Conv2d(8, 16, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(16),
            torch.nn.GELU(),
            # 128
            torch.nn.Conv2d(16, 16, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(16),
            torch.nn.GELU(),
            # 128
        )
        self.layer3 = torch.nn.Sequential(
            # 256
            torch.nn.Conv2d(16, 16, 4, 2, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(16),
            torch.nn.GELU(),
            # 128
            torch.nn.Conv2d(16, 32, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
            # 128
            torch.nn.Conv2d(32, 32, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
            # 128
        )
        self.layer4 = torch.nn.Sequential(
            # 128
            torch.nn.Conv2d(32, 32, 4, 2, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
            # 64
            torch.nn.Conv2d(32, 64, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            # 64
            torch.nn.Conv2d(64, 64, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            # 64
        )
        self.layer5 = torch.nn.Sequential(
            # 64
            torch.nn.Conv2d(64, 64, 4, 2, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            # 32
            torch.nn.Conv2d(64, 128, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            # 32
            torch.nn.Conv2d(128, 128, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            # 32
        )
        self.layer6 = torch.nn.Sequential(
            # 16
            torch.nn.Conv2d(128, 256, 4, 2, 0),
            torch.nn.BatchNorm2d(256),
            torch.nn.GELU(),
            # 7
            torch.nn.Conv2d(256, 256, 3, 1, 0),
            torch.nn.BatchNorm2d(256),
            torch.nn.GELU(),
            # 5
            torch.nn.Upsample((7, 7), mode='bilinear'),
            torch.nn.Conv2d(256, 256, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(256),
            torch.nn.GELU(),
            # 256x7
            torch.nn.Upsample((16, 16), mode='bilinear'),
            torch.nn.Conv2d(256, 128, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            # 128x16
        )
        self.layer7 = torch.nn.Sequential(
            # 32
            torch.nn.Conv2d(128, 128, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            # 32
            torch.nn.Conv2d(128, 128, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            # 128x32
            # torch.nn.PixelShuffle(2),
            torch.nn.Upsample((32, 32), mode='bilinear'),
            torch.nn.Conv2d(128, 64, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            # 32x64
        )
        self.layer8 = torch.nn.Sequential(
            # 64
            torch.nn.Conv2d(64, 64, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            # 64
            torch.nn.Conv2d(64, 64, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            # 64
            # torch.nn.PixelShuffle(2),
            torch.nn.Upsample((64, 64), mode='bilinear'),
            torch.nn.Conv2d(64, 32, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
            # 128
        )
        self.layer9 = torch.nn.Sequential(
            # 16
            torch.nn.Conv2d(32, 32, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
            # 16
            torch.nn.Conv2d(32, 32, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
            # 256x16
            # torch.nn.PixelShuffle(2),
            torch.nn.Upsample((128, 128), mode='bilinear'),
            torch.nn.Conv2d(32, 32, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
            # 32
        )
        self.layer10 = torch.nn.Sequential(
            # 128
            torch.nn.Conv2d(32, 32, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
            # 128
            torch.nn.Conv2d(32, 32, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
            # 128
            torch.nn.Upsample((257, 257), mode='bilinear'),
            torch.nn.Conv2d(32, 16, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(16),
            torch.nn.GELU(),
            # 257
            torch.nn.Conv2d(16, 8, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(8),
            torch.nn.GELU(),
            # 257
            torch.nn.Conv2d(8, 2, 3, 1, 1, padding_mode='replicate'),
            torch.nn.Tanh()
            # 257
        )

        torch.nn.init.uniform_(self.layer10[-2].weight, 0, .01)

    def forward(self, v, H, A, v_a):
        x1 = self.layer1(torch.cat((v, v_a), 1))
        # x1 = torch.cat((x, H[:, None, ...], A[:, None, ...]), 1)
        # x1 = self.layer1(v)
        x2 = self.layer2(torch.cat((x1, H[:, None, ...], A[:, None, ...]), 1))
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6 + x5)  # + x5)
        x8 = self.layer8(x7 + x4)  # + x4)
        x9 = self.layer9(x8 + x3)  # + x3)
        dv = self.layer10(x9)  # + x2)

        return dv
