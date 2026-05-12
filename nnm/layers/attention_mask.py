import torch


def make_causal_mask(seq_len, kv_len, cache_len, device, dtype, *, attn_mask=None, sliding_window=None):
    with torch.no_grad():
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        col_idx = torch.arange(kv_len, device=device).unsqueeze(0)
        mask = torch.where(col_idx <= cache_len + row_idx, 0.0, float('-inf'))

        if sliding_window is not None and kv_len > sliding_window:
            query_position = cache_len + row_idx
            sliding_mask = (query_position - sliding_window) >= col_idx
            mask = mask.masked_fill(sliding_mask, float('-inf'))

        mask = mask.unsqueeze(0).unsqueeze(0)

        if attn_mask is not None:
            batch = attn_mask.shape[0]
            dtype_info = torch.finfo if mask.dtype.is_floating_point else torch.iinfo
            inf_val = dtype_info(mask.dtype).min
            mask = mask.expand(batch, -1, -1, -1).clone()
            padding_mask = (mask + attn_mask[:, None, None, :]) == 0
            mask = mask.masked_fill(padding_mask, inf_val)

        return mask