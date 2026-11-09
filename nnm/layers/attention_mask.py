import torch


@torch.no_grad()
def make_causal_mask(seq_len, kv_len, cache_len, device, dtype, *, attn_mask=None, sliding_window=None, causal=True):
    inf_val = float("-inf")
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    if causal:
        mask = torch.where(col_idx <= cache_len + row_idx, 0.0, inf_val)
    else:
        mask = torch.zeros(seq_len, kv_len, device=device)

    if sliding_window is not None and kv_len > sliding_window:
        query_position = cache_len + row_idx
        if causal:
            sliding_mask = (query_position - sliding_window) >= col_idx
        else:
            sliding_mask = (query_position - col_idx).abs() > sliding_window
        mask = mask.masked_fill(sliding_mask, inf_val)

    mask = mask.unsqueeze(0).unsqueeze(0)

    if attn_mask is not None:
        batch = attn_mask.shape[0]
        mask = mask.expand(batch, -1, -1, -1).clone()
        padding_mask = (mask + attn_mask[:, None, None, :]) == 0
        mask = mask.masked_fill(padding_mask, inf_val)

    return mask
