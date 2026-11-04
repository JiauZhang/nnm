"""BPE tokenizer loaded from the standard ``tokenizer.json`` shipped with Whistle.

The upstream engine stores its vocabulary in a compact binary blob.  It is
equivalent to a HuggingFace BPE tokenizer (vocab + merge priorities + special
tokens + byte fallback), so the weight repository carries ``tokenizer.json``
and this class reproduces the engine's encode/decode without extra dependencies.
"""

import json
import re
from pathlib import Path
from typing import ClassVar

BYTE_TOKEN = re.compile(r"<0x[0-9A-Fa-f]{2}>")
CONTROL = ("<pad>", "</s>", "<s>")


class WhistleTokenizer:
    TYPES: ClassVar[dict[str, int]] = {"normal": 0, "unknown": 1, "control": 2, "user_defined": 3, "byte": 4}

    def __init__(self, source):
        data = source if isinstance(source, dict) else json.loads(Path(source).read_text())
        model = data["model"]
        if model["type"] != "BPE":
            raise ValueError(f"expected a BPE tokenizer, got {model['type']!r}")

        self.piece_to_id = dict(model["vocab"])
        self.pieces = [None] * len(self.piece_to_id)
        for piece, index in self.piece_to_id.items():
            self.pieces[index] = piece

        added = {item["content"]: item for item in data.get("added_tokens", [])}
        self.unk_id = self.piece_to_id[model["unk_token"]]
        self.byte_fallback = bool(model.get("byte_fallback", False))
        self.pad_id = self.piece_to_id["<pad>"]
        self.eos_id = self.piece_to_id["</s>"]
        self.bos_id = self.piece_to_id["<s>"]

        self.types = [self.TYPES["normal"]] * len(self.pieces)
        self.types[self.unk_id] = self.TYPES["unknown"]
        for name in CONTROL:
            if name in self.piece_to_id:
                self.types[self.piece_to_id[name]] = self.TYPES["control"]
        self.byte_to_id = {}
        for piece, index in self.piece_to_id.items():
            if BYTE_TOKEN.fullmatch(piece):
                self.types[index] = self.TYPES["byte"]
                self.byte_to_id[int(piece[3:5], 16)] = index
        for piece in added:
            index = self.piece_to_id[piece]
            if self.types[index] == self.TYPES["normal"]:
                self.types[index] = self.TYPES["user_defined"]
        self.markers = sorted(
            (piece for piece in added if self.types[self.piece_to_id[piece]] == self.TYPES["user_defined"]),
            key=len,
            reverse=True,
        )

        self.scores = [0.0] * len(self.pieces)
        priorities = {}
        for rank, merge in enumerate(model.get("merges", [])):
            left, separator, right = merge.partition(" ")
            if not separator:
                raise ValueError(f"invalid merge {merge!r}")
            piece = left + right
            index = self.piece_to_id.get(piece)
            if index is not None and piece not in priorities:
                priorities[piece] = rank
        for piece, rank in priorities.items():
            self.scores[self.piece_to_id[piece]] = -float(rank)

        pre_tokenizer = data.get("pre_tokenizer") or {}
        scheme = pre_tokenizer.get("prepend_scheme", "never")
        self.add_dummy_prefix = pre_tokenizer.get("type") == "Metaspace" and scheme in ("always", "first")

    def __len__(self):
        return len(self.pieces)

    def _bpe(self, segment):
        symbols = list(segment)
        while len(symbols) > 1:
            best_score, best_index = None, -1
            for index in range(len(symbols) - 1):
                piece_id = self.piece_to_id.get(symbols[index] + symbols[index + 1])
                if piece_id is not None and (best_score is None or self.scores[piece_id] > best_score):
                    best_score, best_index = self.scores[piece_id], index
            if best_index < 0:
                break
            symbols[best_index : best_index + 2] = [symbols[best_index] + symbols[best_index + 1]]

        ids = []
        for symbol in symbols:
            piece_id = self.piece_to_id.get(symbol)
            if piece_id is not None:
                ids.append(piece_id)
            elif self.byte_fallback:
                ids.extend(self.byte_to_id[byte] for byte in symbol.encode("utf-8"))
            else:
                ids.append(self.unk_id)
        return ids

    def encode(self, text):
        if not text:
            return []
        escaped = text.replace(" ", "\u2581")
        if self.add_dummy_prefix:
            escaped = "\u2581" + escaped
        ids, buffer, index, length = [], [], 0, len(escaped)
        while index < length:
            marker = next((candidate for candidate in self.markers if escaped.startswith(candidate, index)), None)
            if marker is not None:
                ids += self._bpe("".join(buffer))
                buffer = []
                ids.append(self.piece_to_id[marker])
                index += len(marker)
            else:
                buffer.append(escaped[index])
                index += 1
        ids += self._bpe("".join(buffer))
        return ids

    def decode(self, ids):
        buffer = bytearray()
        for token in ids:
            kind = self.types[token]
            if kind == self.TYPES["byte"]:
                buffer.append(int(self.pieces[token][3:5], 16))
            elif kind in (self.TYPES["control"], self.TYPES["unknown"]):
                continue
            else:
                buffer += self.pieces[token].encode("utf-8")
        text = buffer.decode("utf-8", "replace").replace("\u2581", " ")
        return text[1:] if self.add_dummy_prefix and text.startswith(" ") else text
