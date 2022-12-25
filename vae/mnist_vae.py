import torch, torchvision
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import numpy as np
import cv2

# https://github.com/cdoersch/vae_tutorial/blob/master/mnist_vae.prototxt
class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.mu = nn.Linear(128, output_dim)
        self.logsd = nn.Linear(128, output_dim)

    def forward(self, inputs):
        output = self.model(inputs)
        z_mean = self.mu(output)
        z_log_var = self.logsd(output)
        epsilon = torch.randn_like(z_mean)
        return z_mean, z_log_var, z_mean + torch.exp(0.5 * z_log_var) * epsilon

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 256),
            nn.Linear(256, output_dim),
        )

    def forward(self, inputs):
        return self.model(inputs)

class VAE():
    def __init__(self, epoches=1000, batch_size=32, sample_interval=1):
        self.encoder = Encoder(input_dim=28*28, output_dim=30)
        self.decoder = Decoder(input_dim=30, output_dim=28*28)

        self.epoches = epoches
        self.batch_size = batch_size
        self.sample_interval = sample_interval
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}], lr=2e-4)

    def train(self):
        # load training dataset
        mnist = torchvision.datasets.MNIST(
            root='.', train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]))
        train_loader = DataLoader(mnist, batch_size=self.batch_size, shuffle=True, drop_last=True)
        for epoch in range(self.epoches):
            train_data_iter = iter(train_loader)
            for step, (images, _) in enumerate(train_data_iter):
                images = images.reshape(self.batch_size, -1)

                z_mean, z_log_var, latent = self.encoder(images)
                rec_images = self.decoder(latent)

                rec_loss = torch.mean(torch.abs(rec_images - images))
                kl_loss = 1 + z_log_var * 2 - torch.square(z_mean) - torch.exp(z_log_var * 2)
                kl_loss = torch.sum(kl_loss, axis=-1)
                kl_loss *= -0.5
                kl_loss = torch.mean(kl_loss)
                vae_loss = rec_loss + kl_loss

                self.optimizer.zero_grad()
                vae_loss.backward()
                self.optimizer.step()

                if step % 50 == 0:
                    print(epoch, step, vae_loss.item())

            if epoch % self.sample_interval == 0:
                rec_images = rec_images.detach().numpy().reshape(-1, 28, 28)
                sample_row = 8
                sample_col = 8
                samples = np.zeros((sample_row * 28, sample_col * 28))
                for r in range(sample_row):
                    for c in range(sample_col):
                        samples[r*28:(r+1)*28, c*28:(c+1)*28] = rec_images[r*sample_col+c]
                sample_name = '{}.png'.format(step)
                cv2.imwrite(sample_name, samples)


if __name__ == '__main__':
    print('\nStart to train VAE...\n')
    vae = VAE(epoches=500, batch_size=64, sample_interval=1)
    vae.train()
    print('\nTraining VAE finished!\n')