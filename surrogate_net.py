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
