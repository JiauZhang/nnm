import unicodedata

MIRROR = dict(zip("()[]{}<>«»‹›", ")(][}{><»«›‹"))
PRESENTATION = ((0xFB1D, 0xFDFF), (0xFE70, 0xFEFF))
TATWEEL = "ـ"


def is_rtl(character):
    return unicodedata.bidirectional(character) in ("R", "AL")


def has_rtl(text):
    return any(is_rtl(character) for character in text)


def rtl_base(text):
    right = sum(is_rtl(character) for character in text)
    left = sum(unicodedata.bidirectional(character) == "L" for character in text)
    return right > left


def normalize(text):
    out = []
    for character in text:
        code = ord(character)
        if any(low <= code <= high for low, high in PRESENTATION):
            out.append(unicodedata.normalize("NFKC", character))
        elif character != TATWEEL and (out or unicodedata.category(character) not in ("Mn", "Me")):
            out.append(character)
    return "".join(out)


def clusters(text):
    out = []
    for character in text:
        if out and unicodedata.category(character) in ("Mn", "Me"):
            out[-1] += character
        else:
            out.append(character)
    return out


def _kind(cluster):
    character = cluster[0]
    return "R" if is_rtl(character) else "L" if unicodedata.bidirectional(character) in ("L", "EN", "AN") else "N"


def _runs(kinds, strong, stop):
    spans, start, last = [], None, None
    for index, kind in enumerate(kinds + ["#"]):
        if kind == strong:
            if start is None:
                start = index
            last = index
        elif kind in stop or kind == "#":
            if start is not None:
                spans.append((start, last + 1))
            start = last = None
    return spans


def _mirror(cluster):
    return MIRROR.get(cluster[0], cluster[0]) + cluster[1:]


def flip(text, mirror=True, base_rtl=None):
    if not has_rtl(text):
        return text
    parts = clusters(text)
    kinds = [_kind(part) for part in parts]
    if rtl_base(text) if base_rtl is None else base_rtl:
        units = []
        position = 0
        for start, end in _runs(kinds, "L", {"R"}) + [(len(parts), len(parts))]:
            for index in range(position, start):
                units.append(_mirror(parts[index]) if mirror else parts[index])
            if start < end:
                units.append("".join(parts[start:end]))
            position = end
        return "".join(reversed(units))
    out = list(parts)
    for start, end in _runs(kinds, "R", {"L"}):
        segment = parts[start:end][::-1]
        out[start:end] = [_mirror(cluster) for cluster in segment] if mirror else segment
    return "".join(out)


def visual_label(logical, base_rtl=None):
    return flip(normalize(logical), base_rtl=base_rtl)


def label_from_visual(visual, shapes=False):
    return visual_label(flip(visual, mirror=shapes))
