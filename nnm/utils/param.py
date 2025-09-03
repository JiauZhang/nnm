import numpy as np
from collections import OrderedDict

def count_parameter(model, precision=None, base=None):
    params = model.parameters()
    count = sum([np.prod(p.size()) for p in params])
    if precision:
        count = format_count(count, precision=precision, base=base)
    else:
        count = int(count)
    return count

def format_count(count, base=None, precision=2):
    base_map = OrderedDict(B=1e9, M=1e6, K=1e3)

    if base not in base_map:
        base = None
        for key in base_map:
            if count / base_map[key] >= 1.0:
                base = key
                break
    if base:
        div = base_map[base]
        value = count / div
    else:
        return str(count)

    fmt = '{:.' + f'{precision}' + 'f}'
    ret = fmt.format(value) + base
    return ret
