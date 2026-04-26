import torch
from nnm.models.magika import Magika

@torch.no_grad()
def test_lfm2_pretrained(model_path):
    state_dict = torch.load(model_path)
    model = Magika()
    model.load_state_dict(state_dict)
