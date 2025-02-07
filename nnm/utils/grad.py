from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def clip_grad_norm(*, parameters, clip_norm, norm_type=2.0):
    return clip_grad_norm_(parameters, clip_norm, norm_type=norm_type)

def clip_grad_value(*, parameters, clip_value):
    clip_grad_value_(parameters, clip_value)
