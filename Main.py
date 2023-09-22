# -*- coding: utf-8 -*-
# @Time    : 2022/12/8 21:04
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12


import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from net.ae import AE
from net.vae import VAE

DEVICE = torch.device('cuda')


DRAW_VISION = False


if DRAW_VISION:
    import visdom
    viz = visdom.Visdom()


def Train(mnist_train, mnist_test):
    # model = AE().to(DEVICE)
    model = VAE().to(DEVICE)

    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # print(model)

    for epoch in range(1000):
        for batchidx, (x, _) in enumerate(mnist_train):
            # [b, 1, 28, 28]
            x = x.to(DEVICE)

            x_hat, kld = model(x)
            loss = criteon(x_hat, x)

            if kld is not None:
                elbo = - loss - 1.0 * kld
                loss = - elbo

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss:', loss.item(), 'kld:', kld.item())

        for batchidx, (x, _) in enumerate(mnist_test):
            x = x.to(DEVICE)
            with torch.no_grad():
                x_hat, kld = model(x)
            if DRAW_VISION:
                viz.images(x, nrow=8, win='x', opts=dict(title='x'))
                viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))


if __name__ == '__main__':
    os.makedirs("dataset", exist_ok=True)

    mnist_train_db = datasets.MNIST('dataset', train=True,
                                    transform=transforms.Compose([transforms.ToTensor()]),
                                    download=True)

    mnist_test_db = datasets.MNIST('dataset', train=False,
                                   transform=transforms.Compose([transforms.ToTensor()]),
                                   download=True)

    mnist_train_loader = DataLoader(mnist_train_db, batch_size=512, shuffle=True)
    mnist_test_loader = DataLoader(mnist_test_db, batch_size=512, shuffle=True)

    Train(mnist_train_loader, mnist_test_loader)
