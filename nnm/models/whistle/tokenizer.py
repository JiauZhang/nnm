import struct


class WhistleTokenizer:
    TYPES = {"normal": 0, "unknown": 1, "control": 2, "user_defined": 3, "byte": 4}
    _HDR = "<IIIIIBBH"
    _REC = "<fBH"

    def __init__(self, blob):
        n, pad, eos, bos, unk, add_dummy, byte_fallback, _ = struct.unpack_from(self._HDR, blob, 0)
        offset = struct.calcsize(self._HDR)
        record = struct.calcsize(self._REC)
        pieces, scores, kinds = [], [], []
        for _ in range(n):
            score, kind, length = struct.unpack_from(self._REC, blob, offset)
            offset += record
            pieces.append(blob[offset : offset + length].decode("utf-8"))
            offset += length
            scores.append(score)
            kinds.append(kind)
        self.pieces, self.scores, self.types = pieces, scores, kinds
        self.pad_id, self.eos_id, self.bos_id, self.unk_id = pad, eos, bos, unk
        self.add_dummy_prefix, self.byte_fallback = bool(add_dummy), bool(byte_fallback)
        self.piece_to_id = {piece: index for index, piece in enumerate(pieces)}
        self.byte_to_id = {
            int(piece[3:5], 16): index for index, (piece, kind) in enumerate(zip(pieces, kinds)) if kind == self.TYPES["byte"]
        }
        self.markers = sorted(
            (piece for piece, kind in zip(pieces, kinds) if kind == self.TYPES["user_defined"]), key=len, reverse=True
        )

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
