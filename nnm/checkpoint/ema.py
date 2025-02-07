import torch

class EMA():
    def __init__(self, model, device='cpu', decay=0.999):
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow_state_dict = {}
        self.load_state_dict(self.model.state_dict())

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        self.shadow_state_dict = {}
        for k, v in state_dict.items():
            self.shadow_state_dict[k] = v.detach().clone().to(self.device)

    def state_dict(self):
        return self.shadow_state_dict

    @torch.no_grad()
    def update(self):
        for k, v in self.model.state_dict().items():
            v_ = v.detach().to(self.device)
            v_ = torch.lerp(v_, self.shadow_state_dict[k], self.decay)
            self.shadow_state_dict[k] = v_
