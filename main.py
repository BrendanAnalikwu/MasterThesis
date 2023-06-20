import torch
import torch.utils.data
from tqdm import trange
from loss import loss_func


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
                                          torch.nn.Tanhshrink(),
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


def advect(v: torch.Tensor, H_old: torch.tensor):
    dH = torch.zeros_like(H_old)

    # x-direction
    v_x = (v[:, 0, 1:, 1:-1] + v[:, 0, :-1, 1:-1]) / 2.
    dH[:, :, 1:] += torch.nn.functional.relu(v_x) * H_old[:, :, :-1]
    dH[:, :, :-1] -= torch.nn.functional.relu(v_x) * H_old[:, :, :-1]
    dH[:, :, :-1] += torch.nn.functional.relu(-1 * v_x) * H_old[:, :, 1:]
    dH[:, :, 1:] -= torch.nn.functional.relu(-1 * v_x) * H_old[:, :, 1:]

    # y-direction
    v_y = (v[:, 0, 1:-1, 1:] + v[:, 0, 1:-1, :-1]) / 2.
    dH[:, 1:, :] += torch.nn.functional.relu(-1 * v_y) * H_old[:, :-1, :]
    dH[:, :-1, :] -= torch.nn.functional.relu(-1 * v_y) * H_old[:, :-1, :]
    dH[:, :-1, :] += torch.nn.functional.relu(v_y) * H_old[:, 1:, :]
    dH[:, 1:, :] -= torch.nn.functional.relu(v_y) * H_old[:, 1:, :]

    return H_old + dH


# Constants
L = 1e6
T = 1e3
G = 1.
C_o = 1026 * 5.5e-3 * L / (900 * G)
C_a = 1.3 * 1.2e-3 * L / (900 * G)
C_r = 27.5e3 * T ** 2 / (2 * 900 * L ** 2)
e_2 = .5
C = 20
f_c = 1.46e-4
dx = 512e3 / 128 / L  # 4km
dt = 1e3 / T  # 1000s

# Initial conditions
N = 4
x, y = torch.meshgrid(torch.linspace(0, 1, 129), torch.linspace(0, 1, 129), indexing='xy')
x_h, y_h = torch.meshgrid(torch.linspace(0, 1 - 1 / 128., 128), torch.linspace(0, 1 - 1 / 128., 128), indexing='xy')
H0 = .3 * torch.ones(N, 128, 128) + .005 * (torch.sin(x_h * 256) + torch.sin(y_h * 256))
A0 = torch.ones(N, 128, 128)
v0 = torch.zeros(N, 2, 129, 129)
v0[:, :, 1:-1, 1:-1] = .01 * torch.ones(N, 2, 127, 127)
v0[:2, 0, :64, :] = -v0[:2, 0, :64, :]
v0[2:, 0, :, :64] = -v0[2:, 0, :, :64]

# Setup NN
model = SurrogateNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# v, H, A = v0, H0, A0
v_a = torch.ones_like(v0) * (20 * T / L)
v_a[::2, 0, :64, :] = -v_a[::2, 0, :64, :]
losses = []

H = advect(v0, H0)
H.clamp_(0., None)
A = advect(v0, A0)
A.clamp_(0., 1.)

for i in trange(10000):
    # v_old = v.detach()
    v = model(v0, H0, A0, v_a)
    loss = loss_func(v, H, A, v0, v_a, C_r, C_a, T, e_2, C, f_c, dx, dt)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    if i == 1000:
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
    optimizer.step()
    losses.append(loss.item())

print('done')
