import torch


class PatchNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer1 = torch.nn.Conv2d(6, 6, 2, 1, 0, bias=False)  # -> 6x4x4
        self.layer2 = torch.nn.Sequential(
            # 8x4x4
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 8, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(8),
            torch.nn.GELU(),
            # 8x4x4
            torch.nn.Conv2d(8, 16, 2, 1, 0),
            torch.nn.BatchNorm2d(16),
            torch.nn.GELU(),
            # 16x3x3
            # torch.nn.Conv2d(16, 16, 3, 1, 1, padding_mode='replicate'),
            # torch.nn.BatchNorm2d(16),
            # torch.nn.GELU(),
            # 16x3x3
            torch.nn.Conv2d(16, 32, 2, 1, 0),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
            # 32x2x2
            # torch.nn.Conv2d(32, 32, 3, 1, 1, padding_mode='replicate'),
            # torch.nn.BatchNorm2d(32),
            # torch.nn.GELU(),
            # 32x2x2
            torch.nn.Conv2d(32, 64, 2, 1, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            # 64x1x1
        )
        # self.layer3 = torch.nn.Sequential(
        #     torch.nn.Linear(64, 32),
        #     torch.nn.BatchNorm1d(32),
        #     torch.nn.GELU())
        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(65, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.GELU()
        )
        self.layer5 = torch.nn.Sequential(
            # 64x1x1
            torch.nn.ConvTranspose2d(64, 32, 2, 1, 0, groups=8),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
            # 32x2x2
            # torch.nn.Conv2d(32, 32, 3, 1, 1, padding_mode='replicate'),
            # torch.nn.BatchNorm2d(32),
            # torch.nn.GELU(),
            # 32x2x2
            torch.nn.Upsample((3, 3), mode='bilinear'),
            torch.nn.Conv2d(32, 16, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(16),
            torch.nn.GELU(),
            # 16x3x3
            # torch.nn.Conv2d(16, 16, 3, 1, 1, padding_mode='replicate'),
            # torch.nn.BatchNorm2d(16),
            # torch.nn.GELU(),
            # 16x3x3
            torch.nn.Upsample((4, 4), mode='bilinear'),
            torch.nn.Conv2d(16, 8, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(8),
            torch.nn.GELU(),
            # 8x4x4
            # torch.nn.Conv2d(8, 8, 3, 1, 1, padding_mode='replicate'),
            # torch.nn.BatchNorm2d(8),
            # torch.nn.GELU(),
            # 8x4x4
            torch.nn.Upsample((5, 5), mode='bilinear'),
            torch.nn.Conv2d(8, 8, 3, 1, 1, padding_mode='replicate'),
            torch.nn.BatchNorm2d(8),
            torch.nn.GELU(),
            # 8x5x5
            torch.nn.Conv2d(8, 2, 3, 1, 1, padding_mode='replicate'),
            torch.nn.Tanhshrink()
            # 2x5x5
        )

    def forward(self, v, H, A, v_a, v_o, border):
        x1 = self.layer1(torch.cat((v, v_a, v_o), 1))
        x2 = self.layer2(torch.cat((x1, H, A), 1))
        # x3 = self.layer3(x2.reshape(-1, 64))
        x4 = self.layer4(torch.cat((x2.reshape(-1, 64), border), 1))
        x5 = self.layer5(x4.reshape(-1, 64, 1, 1))
        return x5


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
