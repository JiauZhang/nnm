import torch
from torch import nn
from torch.nn import functional as F


class EuclideanCodebook(nn.Module):
    def __init__(self, dim, codebook_size, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.cluster_usage = nn.Parameter(torch.ones(codebook_size))
        self.embedding_sum = nn.Parameter(torch.zeros(codebook_size, dim))

    def decode(self, codes):
        embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
        return F.embedding(codes, embedding)

    def encode(self, latent):
        embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
        flat = latent.reshape(-1, latent.shape[-1])
        distances = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ embedding.t() + embedding.pow(2).sum(1)
        return distances.argmin(1).reshape(latent.shape[:-1])


class VectorQuantizer(nn.Module):
    def __init__(self, dim, codebook_size, codebook_dim=None, epsilon=1e-5):
        super().__init__()
        codebook_dim = dim if codebook_dim is None else codebook_dim
        self.project_in = nn.Linear(dim, codebook_dim) if codebook_dim != dim else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if codebook_dim != dim else nn.Identity()
        self._codebook = EuclideanCodebook(codebook_dim, codebook_size, epsilon)

    def decode(self, codes):
        return self.project_out(self._codebook.decode(codes)).transpose(1, 2)

    def encode(self, latent):
        return self._codebook.encode(self.project_in(latent.transpose(1, 2)))


class ResidualVectorQuantization(nn.Module):
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantizer(**kwargs) for _ in range(num_quantizers)])

    def decode(self, codes):
        quantized = torch.zeros([1], device=codes.device)[0]
        for index, layer_codes in enumerate(codes):
            quantized = quantized + self.layers[index].decode(layer_codes)
        return quantized


class BitwiseResidualVQ(nn.Module):
    def __init__(self, dim, num_quantizers, codebook_dim=None):
        super().__init__()
        codebook_dim = dim if codebook_dim is None else codebook_dim
        self.num_quantizers = num_quantizers
        self.codebook_dim = codebook_dim
        self.project_in = nn.Linear(dim, codebook_dim, bias=False) if codebook_dim != dim else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim, bias=False) if codebook_dim != dim else nn.Identity()
        self.codebook = nn.Parameter(torch.zeros(num_quantizers, 2, codebook_dim))

    def decode(self, codes):
        quantizers = torch.arange(self.num_quantizers, device=codes.device)
        embedding = self.codebook[quantizers[None, :], codes].sum(dim=1)
        return self.project_out(embedding)
