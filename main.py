import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from tqdm import trange


class AutoEncoderNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = torch.nn.Sequential(
            # Input: 1x28
            torch.nn.Conv2d(1, 4, 4, 2, 0),
            torch.nn.LeakyReLU(.1),
            # 4x13
            torch.nn.Conv2d(4, 8, 3, 2, 0),
            torch.nn.LeakyReLU(.1),
            # 8x6
            torch.nn.Conv2d(8, 16, 3, 1, 0),
            torch.nn.LeakyReLU(.1),
            # 16x4
            torch.nn.Flatten(1, 3),
            # Nx256
            torch.nn.Linear(256, 64),
            torch.nn.LeakyReLU(.1),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(.1),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(.1),
            # Nx16
        )

        self.decoder = torch.nn.Sequential(
            # Input: Nx16
            torch.nn.Linear(16, 32),
            torch.nn.LeakyReLU(.1),
            torch.nn.Linear(32, 64),
            torch.nn.LeakyReLU(.1),
            torch.nn.Linear(64, 256),
            torch.nn.LeakyReLU(.1),
            # Nx256
            torch.nn.Unflatten(1, (16, 4, 4)),
            # Nx16x4x4
            torch.nn.ConvTranspose2d(16, 8, 3, 1, 1),
            torch.nn.ConvTranspose2d(8, 8, 3, 1, 0),
            torch.nn.ReLU(),
            # 8x6
            torch.nn.ConvTranspose2d(8, 4, 3, 1, 1),
            torch.nn.ConvTranspose2d(4, 4, 3, 2, 0),
            torch.nn.ReLU(),
            # 4x13
            torch.nn.ConvTranspose2d(4, 4, 3, 1, 1),
            torch.nn.ConvTranspose2d(4, 4, 4, 2, 0),
            torch.nn.ReLU(),
            # 4x28
            torch.nn.ConvTranspose2d(4, 4, 3, 1, 1),
            torch.nn.ConvTranspose2d(4, 1, 3, 1, 1),
            # 1x28
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Transform image to tensor
tensor_transform = transforms.ToTensor()

# Get data
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=tensor_transform)
dataset_test = datasets.MNIST(root="./data", train=False, download=True, transform=tensor_transform)

loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)
loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=64, shuffle=True)
test, _ = next(iter(loader_test))
save_image(make_grid(test.detach(), nrow=8), f"data_test.png")

model = AutoEncoderNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 20
outputs = []
losses = []
loss_func = torch.nn.MSELoss()
for epoch in trange(epochs):
    for (im, _) in loader:
        res = model(im)
        loss = loss_func(res, im)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    outputs.append((epochs, im.detach(), res.detach()))
    save_image(make_grid(model(test).detach(), nrow=8), f"result_{epoch}.png")

plt.plot(losses)

plt.imshow(transforms.ToPILImage()(make_grid(im.detach(), nrow=4)))
plt.figure()
plt.imshow(transforms.ToPILImage()(make_grid(res.detach(), nrow=4)))

plt.show()
