import torch

class EMA():
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow_state_dict = {}
        self.load_state_dict(self.model.state_dict())

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        self.shadow_state_dict = {}
        for k, v in state_dict.items():
            if v.requires_grad:
                self.shadow_state_dict[k] = v.detach().clone()
            else:
                self.shadow_state_dict[k] = v

    def state_dict(self):
        return self.shadow_state_dict

    @torch.no_grad()
    def update(self):
        for k, v in self.model.state_dict().items():
            if v.requires_grad:
                self.shadow_state_dict[k] = torch.lerp(v, self.shadow_state_dict[k], self.decay)
            else:
                self.shadow_state_dict[k] = v
