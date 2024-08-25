import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
from torchstudio.dataset.mnist import MNISTDataset

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
        z_log_std = self.logsd(output)
        epsilon = torch.randn_like(z_mean)
        return z_mean, z_log_std, z_mean + torch.exp(z_log_std) * epsilon

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, inputs):
        return torch.sigmoid(self.model(inputs))

class VAE():
    def __init__(self, latent_dim=30, epoches=1000, batch_size=32, sample_interval=1):
        self.encoder = Encoder(input_dim=28*28, output_dim=latent_dim)
        self.decoder = Decoder(input_dim=latent_dim, output_dim=28*28)

        self.latent_dim = latent_dim
        self.epoches = epoches
        self.batch_size = batch_size
        self.sample_interval = sample_interval
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}], lr=2e-4)
        self.z = torch.randn((self.batch_size, self.latent_dim))

    def train(self):
        mnist = MNISTDataset(root='.')
        train_loader = DataLoader(mnist, batch_size=self.batch_size, shuffle=True, drop_last=True)
        for epoch in range(self.epoches):
            train_data_iter = iter(train_loader)
            for step, (images, _) in enumerate(train_data_iter):
                images = images.reshape(self.batch_size, -1)

                z_mean, z_log_std, latent = self.encoder(images)
                rec_images = self.decoder(latent)

                rec_loss = nn.functional.binary_cross_entropy(rec_images, images, reduction='sum')
                kl_loss = 1 + z_log_std * 2 - torch.square(z_mean) - torch.exp(z_log_std * 2)
                kl_loss = torch.sum(kl_loss)
                kl_loss *= -0.5
                vae_loss = rec_loss + kl_loss

                self.optimizer.zero_grad()
                vae_loss.backward()
                self.optimizer.step()

                if step % 50 == 0:
                    print(epoch, step, vae_loss.item())

            if epoch % self.sample_interval == 0:
                rec_images = rec_images.reshape(-1, 1, 28, 28)
                save_image(rec_images, f'o-{epoch}.png')
                images = images.view(-1, 1, 28, 28)
                save_image(images, f'i-{epoch}.png')
                fake_images = self.decoder(self.z).view(-1, 1, 28, 28)
                save_image(fake_images, f'g-{epoch}.png')

if __name__ == '__main__':
    print('\nStart to train VAE...\n')
    vae = VAE(epoches=50, batch_size=64, sample_interval=1)
    vae.train()
    print('\nTraining VAE finished!\n')