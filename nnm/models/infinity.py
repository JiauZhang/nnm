from torch import nn

from nnm.layers.dit import DiTBlock
from nnm.layers.vq import BitwiseResidualVQ
from nnm.models.pretrained import PretrainedModel


class BitwiseClassifier(nn.Module):
    def __init__(self, hidden_size, num_quantizers, num_bits):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_bits = num_bits
        self.proj = nn.Linear(hidden_size, num_quantizers * num_bits, bias=True)

    def forward(self, hidden):
        return self.proj(hidden).reshape(*hidden.shape[:-1], self.num_quantizers, self.num_bits)


class Infinity(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.get("hidden_size", config.get("embed_dim"))
        feature_dim = config.get("feature_dim", hidden_size)
        num_heads = config.get("num_attention_heads", config.get("num_heads"))
        num_layers = config.get("num_hidden_layers", config.get("depth"))
        num_classes = config.get("num_classes", 1000)
        num_scales = config.get("num_scales", 10)
        intermediate_size = config.get("intermediate_size", hidden_size * 4)
        num_quantizers = config.get("num_quantizers", 16)
        num_bits = config.get("num_bits", 1)
        self.x_embedder = nn.Linear(feature_dim, hidden_size, bias=True)
        self.class_embed = nn.Embedding(num_classes, hidden_size)
        self.scale_embed = nn.Embedding(num_scales, hidden_size)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)])
        self.head = BitwiseClassifier(hidden_size, num_quantizers, num_bits)
        self.quantizer = BitwiseResidualVQ(feature_dim, num_quantizers)

    def forward(self, features, class_labels, scale):
        hidden = self.x_embedder(features)
        condition = self.class_embed(class_labels) + self.scale_embed(scale)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.head(hidden)

    def decode(self, codes):
        return self.quantizer.decode(codes)
