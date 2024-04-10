from typing import Tuple

import torch


class PatchNet(torch.nn.Module):
    def __init__(self, overlap: int = 1, out_size: int = 3, n_hidden: Tuple[int, int, int] = (16, 32, 64), complexity=0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert overlap > complexity

        self.out_size = out_size
        self.overlap = overlap
        self.n_hidden = n_hidden
        self.complexity = complexity

        self.input_layer_v = torch.nn.Conv2d(6, n_hidden[0], 3, 1, 0, bias=False)  # -> -2
        self.input_layer_HA = torch.nn.Conv2d(2, n_hidden[0], 2, 1, 0, bias=False)  # -> -1
        self.input_layer_border = torch.nn.Linear(1, n_hidden[0], bias=True)  # -> 16x1x1
        self.input_activation = torch.nn.Sequential(torch.nn.BatchNorm2d(n_hidden[0]), torch.nn.GELU())

        self.encoder = torch.nn.Sequential(torch.nn.Conv2d(n_hidden[0], n_hidden[1], 3, 1, 0),
                                           torch.nn.BatchNorm2d(n_hidden[1]),
                                           torch.nn.GELU(),
                                           *[torch.nn.Conv2d(n_hidden[1], n_hidden[1], 3, 1, 0),
                                           torch.nn.BatchNorm2d(n_hidden[1]),
                                           torch.nn.GELU()] * complexity,
                                           torch.nn.Conv2d(n_hidden[1], n_hidden[1], 2 * overlap - 1, 1, 0),
                                           torch.nn.BatchNorm2d(n_hidden[1]),
                                           torch.nn.GELU(),
                                           torch.nn.Flatten(1, -1),
                                           torch.nn.Linear(n_hidden[1] * (out_size - 2 - 2 * complexity) ** 2, n_hidden[2]),
                                           torch.nn.BatchNorm1d(n_hidden[2]),
                                           torch.nn.GELU())

        dec = []
        for i in range(complexity):
            dec.extend([torch.nn.Upsample((out_size - 2 * complexity + (i + 1) * 2), mode='bilinear'),
                        torch.nn.Conv2d(n_hidden[1], n_hidden[1], 3, 1, 1),
                        torch.nn.BatchNorm2d(n_hidden[1]),
                        torch.nn.GELU()])

        self.decoder = torch.nn.Sequential(torch.nn.Linear(n_hidden[2], n_hidden[1] * (out_size - 2 - 2 * complexity) ** 2),
                                           torch.nn.BatchNorm1d(n_hidden[1] * (out_size - 2 - 2 * complexity) ** 2),
                                           torch.nn.GELU(),
                                           torch.nn.Unflatten(-1, (n_hidden[1], out_size - 2 - 2 * complexity, out_size - 2 - 2 * complexity)),
                                           torch.nn.ConvTranspose2d(n_hidden[1], n_hidden[1], 3, 1, 0),
                                           torch.nn.BatchNorm2d(n_hidden[1]),
                                           torch.nn.GELU(),
                                           *dec,
                                           torch.nn.ConvTranspose2d(n_hidden[1], 2, 3, 1, 1))

    def forward(self, v, H=None, A=None, v_a=None, v_o=None, border=None):
        x = self.input_activation(self.input_layer_v(torch.cat((v, v_a, v_o), 1))
                                  + self.input_layer_HA(torch.cat((H, A), 1))
                                  + self.input_layer_border(border).transpose(1, 3))
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2


class SurrogateNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer_v = torch.nn.Conv2d(6, 16, 3, 1, 0, bias=False)  # -> -2
        self.input_layer_HA = torch.nn.Conv2d(2, 16, 2, 1, 0, bias=False)  # -> -1
        self.input_activation = torch.nn.Sequential(torch.nn.BatchNorm2d(16), torch.nn.GELU())

        self.encoder = torch.nn.Sequential(torch.nn.Conv2d(16, 128, 17, 16, 1),
                                           torch.nn.BatchNorm2d(128),
                                           torch.nn.GELU(),
                                           torch.nn.Conv2d(128, 512, 4, 4, 0),
                                           torch.nn.BatchNorm2d(512),
                                           torch.nn.GELU(),
                                           torch.nn.Conv2d(512, 1024, 4, 1, 0),
                                           torch.nn.BatchNorm2d(1024),
                                           torch.nn.GELU())
        self.bottleneck = torch.nn.Sequential(torch.nn.Flatten(1, -1),
                                              torch.nn.Linear(1024, 1024),
                                              # torch.nn.BatchNorm2d(128),
                                              torch.nn.GELU(),
                                              torch.nn.Unflatten(1, (1024, 1, 1))
                                              )
        self.decoder = torch.nn.Sequential(torch.nn.ConvTranspose2d(1024, 512, 8, 1, 0),
                                           torch.nn.BatchNorm2d(512),
                                           torch.nn.GELU(),
                                           torch.nn.ConvTranspose2d(512, 256, 8, 8, 0),
                                           torch.nn.BatchNorm2d(256),
                                           torch.nn.GELU(),
                                           torch.nn.ConvTranspose2d(256, 32, 2, 2, 0),
                                           torch.nn.BatchNorm2d(32),
                                           torch.nn.GELU(),
                                           torch.nn.ConvTranspose2d(32, 2, 3, 2, 0)
                                           )

    def forward(self,  v, H, A, v_a, v_o):
        x = self.input_activation(self.input_layer_v(torch.cat((v, v_a, v_o), 1))
                                  + self.input_layer_HA(torch.cat((H, A), 1)))
        x1 = self.encoder(x)
        x2 = self.bottleneck(x1)
        x3 = self.decoder(x2)
        return x3


class SmallSurrogateNet(SurrogateNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer_v = torch.nn.Conv2d(6, 8, 3, 1, 0, bias=False)  # -> -2
        self.input_layer_HA = torch.nn.Conv2d(2, 8, 2, 1, 0, bias=False)  # -> -1
        self.input_activation = torch.nn.Sequential(torch.nn.BatchNorm2d(8), torch.nn.GELU())

        self.encoder = torch.nn.Sequential(torch.nn.Conv2d(8, 32, 17, 16, 1),
                                           torch.nn.BatchNorm2d(32),
                                           torch.nn.GELU(),
                                           torch.nn.Conv2d(32, 96, 4, 4, 0),
                                           torch.nn.BatchNorm2d(96),
                                           torch.nn.GELU(),
                                           torch.nn.Conv2d(96, 256, 4, 1, 0),
                                           torch.nn.BatchNorm2d(256),
                                           torch.nn.GELU())
        self.bottleneck = torch.nn.Sequential(torch.nn.Flatten(1, -1),
                                              torch.nn.Linear(256, 256),
                                              # torch.nn.BatchNorm2d(128),
                                              torch.nn.GELU(),
                                              torch.nn.Unflatten(1, (256, 1, 1))
                                              )
        self.decoder = torch.nn.Sequential(torch.nn.ConvTranspose2d(256, 96, 8, 1, 0),
                                           torch.nn.BatchNorm2d(96),
                                           torch.nn.GELU(),
                                           torch.nn.ConvTranspose2d(96, 32, 8, 8, 0),
                                           torch.nn.BatchNorm2d(32),
                                           torch.nn.GELU(),
                                           torch.nn.ConvTranspose2d(32, 8, 2, 2, 0),
                                           torch.nn.BatchNorm2d(8),
                                           torch.nn.GELU(),
                                           torch.nn.ConvTranspose2d(8, 2, 3, 2, 0)
                                           )


class NoisySurrogateNet(SurrogateNet):
    scale = .1

    def __init__(self, *args, scale=.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale

    def forward(self,  v, H, A, v_a, v_o):
        x = self.input_activation(self.input_layer_v(torch.cat((v, v_a, v_o), 1))
                                  + self.input_layer_HA(torch.cat((H, A), 1)))
        x1 = self.encoder(x)
        if self.training:
            x2 = self.bottleneck(x1) + torch.randn_like(x1) * self.scale**2
        else:
            x2 = self.bottleneck(x1)
        x3 = self.decoder(x2)
        return x3


class UNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_layer_v = torch.nn.Conv2d(4, 64, 7, 4, 1)  # 257 -> 64
        self.input_layer_HA = torch.nn.Conv2d(2, 64, 6, 4, 1)  # 256 -> 64
        self.input_activation = torch.nn.Sequential(torch.nn.BatchNorm2d(64), torch.nn.GELU())
        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1, padding_mode='zeros'),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, 64, 3, 1, 1, padding_mode='zeros'),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU()
        )

        self.layer1 = UNetLayerDown(64, 128, torch.nn.GELU, torch.nn.BatchNorm2d)  # 64 -> 32
        self.layer2 = UNetLayerDown(128, 256, torch.nn.GELU, torch.nn.BatchNorm2d)  # 32 -> 16
        self.layer3 = UNetLayerDown(256, 512, torch.nn.GELU, torch.nn.BatchNorm2d)  # 16 -> 8
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, 4, 2, 1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.GELU(),
            torch.nn.Conv2d(1024, 1024, 4, 1, 0),
            torch.nn.BatchNorm2d(1024),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(1024, 1024, 4, 1, 0),
            torch.nn.BatchNorm2d(1024),
            torch.nn.GELU()
        )  # 8 -> 4 -> 1 -> 4
        self.layer4 = UNetLayerUp(1024, 512, torch.nn.GELU, torch.nn.BatchNorm2d)  # 4 -> 8
        self.layer5 = UNetLayerUp(512, 256, torch.nn.GELU, torch.nn.BatchNorm2d)  # 8 -> 16
        self.layer6 = UNetLayerUp(256, 128, torch.nn.GELU, torch.nn.BatchNorm2d)  # 16 -> 32
        self.layer7 = UNetLayerUp(128, 64, torch.nn.GELU, torch.nn.BatchNorm2d)  # 32 -> 64
        self.output = torch.nn.Sequential(torch.nn.Upsample((257, 257)),  # 64 -> 257
                                          torch.nn.Conv2d(64, 32, 3, 1, 1),
                                          torch.nn.BatchNorm2d(32),
                                          torch.nn.GELU(),
                                          torch.nn.Conv2d(32, 2, 3, 1, 1),
                                          torch.nn.Tanh())
                                          # torch.nn.BatchNorm2d(32),
                                          # torch.nn.GELU(),
                                          # torch.nn.Conv2d(32, 32, 3, 1, 1, padding_mode='zeros'),
                                          # torch.nn.GELU(),
                                          # torch.nn.Conv2d(32, 2, 3, 1, 1, padding_mode='zeros'),
                                          # SymmetricExponentialUnit())  # ,
                                            # torch.nn.InstanceNorm2d(2, affine=False))  # 257 -> 257

    def forward(self,  v, H, A, v_a):
        x = self.input_activation(self.input_layer_v(torch.cat((v, v_a), 1))
                                  + self.input_layer_HA(torch.cat((H, A), 1)))
        x1 = self.input_conv(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.bottleneck(x4)
        x6 = self.layer4(x5, x4)
        x7 = self.layer5(x6, x3)
        x8 = self.layer6(x7, x2)
        x9 = self.layer7(x8, x1)
        out = self.output(x9)
        return out  # torch.nn.functional.pad(out, (1, 1, 1, 1))  # , torch.exp(-(code[:, :2]/10).square()) * 2, code[:, 2:4] / 10

    def encoder(self, v, H, A, v_a):
        x = self.input_activation(self.input_layer_v(torch.cat((v, v_a), 1))
                                  + self.input_layer_HA(torch.cat((H, A), 1)))
        x1 = self.input_conv(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.bottleneck(x4)
        return x1, x2, x3, x4, x5

    def decoder(self, x1, x2, x3, x4, x5):
        x6 = self.layer4(x5, x4)
        x7 = self.layer5(x6, x3)
        x8 = self.layer6(x7, x2)
        x9 = self.layer7(x8, x1)
        out = self.output(x9)
        return out  # torch.nn.functional.pad(out, (1, 1, 1, 1)), x9


class UNetLayerDown(torch.nn.Module):
    def __init__(self, size_in, size_out, activation, normalization, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(size_in, size_out, 4, 2, 1, padding_mode='zeros'),
            normalization(size_out),
            activation(),
            torch.nn.Conv2d(size_out, size_out, 3, 1, 1, padding_mode='zeros'),
            normalization(size_out),
            activation(),
            torch.nn.Conv2d(size_out, size_out, 3, 1, 1, padding_mode='zeros'),
            normalization(size_out),
            activation()
        )

    def forward(self, x: torch.tensor):
        return self.layer(x)


class UNetLayerUp(torch.nn.Module):
    def __init__(self, size_in, size_out, activation, normalization, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(size_in, size_out, 3, 1, 1, padding_mode='zeros'),
            normalization(size_out),
            activation())
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(size_in, size_out, 3, 1, 1, padding_mode='zeros'),
            normalization(size_out),
            activation(),
            torch.nn.Conv2d(size_out, size_out, 3, 1, 1, padding_mode='zeros'),
            normalization(size_out),
            activation()
        )

    def forward(self, x: torch.tensor, skip: torch.tensor):
        x1 = self.layer1(x)
        return self.layer2(torch.cat((x1, skip), dim=1))


class SymmetricExponentialUnit(torch.nn.Module):
    def __init__(self):
        super(SymmetricExponentialUnit, self).__init__()

    def forward(self, x):
        return torch.nn.functional.relu(torch.exp(x) - 1) - torch.nn.functional.relu(torch.exp(-x) - 1) - x
