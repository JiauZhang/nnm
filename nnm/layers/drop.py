import torch

def drop_path(input, drop_prob=0.0, training=False, normalization=True):
    if drop_prob <= 0.0 or training == False:
        return input
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    keep_prob = 1 - drop_prob
    mask = torch.empty(shape).bernoulli_(keep_prob)
    if normalization:
        mask.div_(keep_prob)
    return input * mask
