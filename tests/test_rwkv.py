import torch, pytest
from nnm.layers.rwkv import RWKVTimeMix

# https://github.com/huggingface/transformers/blob/27a25bee4fcb865e8799ba026f1ea4455f2cca98/src/transformers/models/rwkv/modeling_rwkv.py#L189
def hf_rwkv_linear_attention_cpu(time_decay, time_first, key, value, state=None, return_state=False):
    _, seq_length, _ = key.size()
    output = torch.zeros_like(key)

    if state is None:
        num_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
        den_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
        max_state = torch.zeros_like(key[:, 0], dtype=torch.float32) - 1e38
    else:
        num_state, den_state, max_state = state

    time_decay = -torch.exp(time_decay)

    for current_index in range(seq_length):
        current_key = key[:, current_index].float()
        current_value = value[:, current_index]

        # wkv computation at time t
        max_for_output = torch.maximum(max_state, current_key + time_first)
        e1 = torch.exp(max_state - max_for_output)
        e2 = torch.exp(current_key + time_first - max_for_output)
        numerator = e1 * num_state + e2 * current_value
        denominator = e1 * den_state + e2
        output[:, current_index] = (numerator / denominator).to(output.dtype)

        # Update state for next iteration
        max_for_state = torch.maximum(max_state + time_decay, current_key)
        e1 = torch.exp(max_state + time_decay - max_for_state)
        e2 = torch.exp(current_key - max_for_state)
        num_state = e1 * num_state + e2 * current_value
        den_state = e1 * den_state + e2
        max_state = max_for_state

    if return_state or state is not None:
        state = [num_state, den_state, max_state]

    return output, state

@pytest.mark.parametrize('batch, seq_len, embed_dim', [(1, 64, 16), (4, 256, 128)])
def test_rwkv_time_mix(batch, seq_len, embed_dim):
    tm = RWKVTimeMix(256)
    w = torch.randn(1, 1, embed_dim)
    u = torch.randn(1, 1, embed_dim)
    k = torch.randn(batch, seq_len, embed_dim)
    v = torch.randn(batch, seq_len, embed_dim)
    o = tm.wkv(w, u, k, v)
    hf_o, _ = hf_rwkv_linear_attention_cpu(w, u, k, v)

    assert o.shape == hf_o.shape
    assert torch.abs(o - hf_o).mean() < 1e-5