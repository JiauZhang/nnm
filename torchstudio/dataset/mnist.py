import torchvision
from torch.utils.data.dataloader import DataLoader

def get_dataloader(batch_size)
    mnist = torchvision.datasets.MNIST(
        root='.', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ]))
    train_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader