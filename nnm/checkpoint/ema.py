import torch

class EMA():
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.state_dict = {}
        for k, v in self.model.state_dict().items():
            self.state_dict[k] = v.clone()

    @torch.no_grad()
    def update(self):
        for k, v in self.model.state_dict().items():
            self.state_dict[k] = self.state_dict[k] * self.decay + v * (1 - self.decay)
