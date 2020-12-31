import ray

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.srgan import *
from model.loss import *
from utils.preprocessing import ImageDataSet


if __name__ == '__main__':
    import ssl
    import cv2

    import time

    ssl._create_default_https_context = ssl._create_unverified_context

    dataset = ImageDataSet('/Users/andreeaariton/PycharmProjects/super-resolution/data/DIV2K_valid_HR')
    dataloader = DataLoader(
        dataset,
        shuffle=True
    )

    generator = GeneratorNetwork()
    discriminator = DiscriminatorNetwork()

    optim_g = optim.Adam(generator.parameters(), lr=0.001)
    optim_d = optim.Adam(discriminator.parameters(), lr=0.001)

    content_criterion = ContentLoss()
    adversarial_criterion = nn.L1Loss()
    discriminator_criterion = nn.L1Loss()

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(10):
            for i, (lr, hr) in enumerate(dataloader):
                lr = lr.reshape(1, 3, 100, 100)
                hr = hr.reshape(1, 3, 400, 400)

                real, fake = torch.ones((1, 1)), torch.zeros((1, 1))
                real.requires_grad, fake.requires_grad = False, False

                # Generator
                optim_g.zero_grad()

                gen_out = generator(lr)

                content_loss = content_criterion(gen_out, hr)

                adversarial_loss = 1e-3 * adversarial_criterion(discriminator(gen_out), real)

                perceptual_loss = content_loss + adversarial_loss

                perceptual_loss.backward(retain_graph=True)
                optim_g.step()

                # Discriminator
                optim_d.zero_grad()

                real_out = discriminator(hr)
                fake_out = discriminator(gen_out.detach())

                discriminator_loss = 1 - (real_out + fake_out) / 2

                discriminator_loss.backward(retain_graph=True)
                optim_d.step()

                print(f'Iteration: {i}, D_loss: {discriminator_loss.item()}, G_loss: {perceptual_loss}')

            torch.save(generator, 'generator.pth')
            torch.save(discriminator, 'discriminator.pth')
