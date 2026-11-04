import math

import numpy as np

BITS = 11
TAG_SHIFT = 61
BOS = -1


def _pack(tokens, bits=BITS):
    key = 0
    for token in tokens:
        key = (key << bits) | (0 if token == BOS else token + 1)
    return key


class NgramLM:
    def __init__(self, path):
        data = np.load(path)
        self.order = int(data["order"])
        self.vocab_size = int(data["vocab_size"])
        self.bits = int(data["bits"]) if "bits" in data else BITS
        if self.order * self.bits > TAG_SHIFT:
            raise ValueError(f"order {self.order} x {self.bits} bits do not fit a key")

        self.keys = [data[f"k{index}"] for index in range(self.order)]
        self.values = [data[f"v{index}"] for index in range(self.order)]
        self.history_keys = [data[f"hk{index}"] for index in range(self.order)]
        self.history_values = [data[f"hb{index}"] for index in range(self.order)]
        self.cache = {}
        self.key_cache = {}

        tags = [np.uint64(index) << np.uint64(TAG_SHIFT) for index in range(self.order)]
        for name, keys, values in (
            ("p", self.keys, self.values),
            ("h", self.history_keys, self.history_values),
        ):
            combined = np.concatenate([table | tag for table, tag in zip(keys, tags)])
            sorted_values = np.concatenate(values).astype(np.float64)
            order = np.argsort(combined)
            setattr(self, f"{name}_keys", combined[order])
            setattr(self, f"{name}_values", sorted_values[order])

        lengths = np.arange(self.order, dtype=np.uint64)
        self.tags = (lengths << np.uint64(TAG_SHIFT))[None, :]
        self.masks = ((np.uint64(1) << (np.uint64(self.bits) * lengths)) - np.uint64(1))[None, :]

    @property
    def max_logp(self):
        if not hasattr(self, "_max_logp"):
            top = max(float(self.p_values.max()), -math.log(self.vocab_size))
            backoff = sum(max(0.0, float(table.max())) for table in self.history_values if len(table))
            self._max_logp = top + backoff
        return self._max_logp

    @staticmethod
    def _find(keys, key):
        index = int(np.searchsorted(keys, np.uint64(key)))
        return index if index < len(keys) and int(keys[index]) == key else -1

    def logp(self, history, token):
        cache_key = (history, token)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        top = min(len(history), self.order - 1)
        suffixes = [tuple(history[len(history) - length :]) if length else () for length in range(top + 1)]
        seen = 0
        for length in range(1, top + 1):
            if self._find(self.history_keys[length], _pack(suffixes[length], self.bits)) < 0:
                break
            seen = length

        total, value = 0.0, None
        for length in range(seen, -1, -1):
            index = self._find(self.keys[length], _pack(suffixes[length] + (token,), self.bits))
            if index >= 0:
                value = total + float(self.values[length][index])
                break
            total += float(
                self.history_values[length][self._find(self.history_keys[length], _pack(suffixes[length], self.bits))]
            )
        if value is None:
            value = total - math.log(self.vocab_size)
        if len(self.cache) < 1_000_000:
            self.cache[cache_key] = value
        return value

    def logp_keys(self, keys):
        out = [self.key_cache.get(key) for key in keys]
        pending = [index for index, value in enumerate(out) if value is None]
        if not pending:
            return out

        count = len(pending)
        full = np.array([keys[index] for index in pending], np.uint64)[:, None]
        bits = np.uint64(self.bits)
        current = full & np.uint64((1 << self.bits) - 1)
        suffix = (full >> bits) & self.masks
        pair_query = (suffix << bits) | current | self.tags
        history_query = suffix | self.tags

        hits = []
        for table, values, query in (
            (self.h_keys, self.h_values, history_query),
            (self.p_keys, self.p_values, pair_query),
        ):
            position = np.minimum(np.searchsorted(table, query), len(table) - 1)
            hit = table[position] == query
            hits.append((hit, np.where(hit, values[position], 0.0)))
        (history_hit, history_backoff), (pair_hit, pair_value) = hits
        history_hit[:, 0] = True

        seen = np.cumprod(history_hit[:, 1:], axis=1).sum(1) if self.order > 1 else np.zeros(count, int)
        valid = np.arange(self.order)[None, :] <= seen[:, None]
        pair_valid = valid & pair_hit
        found = pair_valid.any(1)
        chosen = self.order - 1 - np.argmax(pair_valid[:, ::-1], axis=1)
        suffix_backoff = np.zeros((count, self.order + 1))
        suffix_backoff[:, : self.order] = np.cumsum((history_backoff * valid)[:, ::-1], axis=1)[:, ::-1]
        rows = np.arange(count)
        result = np.where(
            found,
            suffix_backoff[rows, chosen + 1] + pair_value[rows, chosen],
            suffix_backoff[:, 0] - math.log(self.vocab_size),
        )
        for row, index in enumerate(pending):
            out[index] = float(result[row])
            if len(self.key_cache) < 1_000_000:
                self.key_cache[keys[index]] = out[index]
        return out


def _lse(left, right):
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    top = max(left, right)
    return top + math.log(math.exp(left - top) + math.exp(right - top))


def _add(table, prefix, ending_blank, ending_token, score, steps, history):
    old = table.get(prefix)
    if old is None:
        table[prefix] = (ending_blank, ending_token, score, steps, history)
    else:
        table[prefix] = (_lse(old[0], ending_blank), _lse(old[1], ending_token), *old[2:])


def beam_search(blank, topv, topi, lm, alpha=0.5, beta=1.5, beam=8, blank_skip=0.999, cand_floor=1e-3):
    bits = lm.bits
    history_mask = (1 << (bits * (lm.order - 1))) - 1
    beams = {(): (0.0, -math.inf, 0.0, (), 0)}
    log_skip = math.log(blank_skip)
    log_floor = math.log(cand_floor)

    for step in range(len(blank)):
        blank_logp = float(blank[step])
        if blank_logp > log_skip:
            beams = {
                prefix: (_lse(entry[0], entry[1]) + blank_logp, -math.inf, entry[2], entry[3], entry[4])
                for prefix, entry in beams.items()
            }
            continue

        candidates = [
            (int(identifier), float(value))
            for identifier, value in zip(topi[step], topv[step])
            if identifier != 0 and value >= log_floor
        ]
        nxt = {}

        for prefix, (ending_blank, ending_token, score, steps, history) in beams.items():
            _add(nxt, prefix, _lse(ending_blank, ending_token) + blank_logp, -math.inf, score, steps, history)
            last = prefix[-1] if prefix else None
            for identifier, logp in candidates:
                if identifier == last:
                    _add(nxt, prefix, -math.inf, ending_token + logp, score, steps, history)

        floor = min(_lse(entry[0], entry[1]) + entry[2] for entry in nxt.values()) if len(beams) >= beam else -math.inf
        reach = beta + alpha * lm.max_logp
        extensions = []
        for prefix, (ending_blank, ending_token, score, steps, history) in beams.items():
            total = _lse(ending_blank, ending_token)
            last = prefix[-1] if prefix else None
            for identifier, logp in candidates:
                new_from = ending_blank + logp if identifier == last else total + logp
                if new_from == -math.inf:
                    continue
                extended = prefix + (identifier,)
                if extended in nxt:
                    _add(nxt, extended, -math.inf, new_from, score, steps, history)
                elif new_from + score + reach >= floor - 1e-9:
                    extensions.append((prefix, identifier, new_from, score, steps, history))

        logps = lm.logp_keys([(entry[5] << bits) | (entry[1] + 1) for entry in extensions]) if extensions else []
        for (prefix, identifier, new_from, score, steps, history), logp in zip(extensions, logps):
            _add(
                nxt,
                prefix + (identifier,),
                -math.inf,
                new_from,
                score + alpha * logp + beta,
                steps + (step,),
                ((history << bits) | (identifier + 1)) & history_mask,
            )

        beams = dict(sorted(nxt.items(), key=lambda item: -(_lse(item[1][0], item[1][1]) + item[1][2]))[:beam])

    best = max(beams.items(), key=lambda item: _lse(item[1][0], item[1][1]) + item[1][2])
    return list(best[0]), list(best[1][3])
