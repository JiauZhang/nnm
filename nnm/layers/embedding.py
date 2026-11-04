def merge_placeholders(inputs_embeds, mask, features):
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(inputs_embeds)
    return inputs_embeds.masked_scatter(mask, features.to(inputs_embeds.device, inputs_embeds.dtype))
