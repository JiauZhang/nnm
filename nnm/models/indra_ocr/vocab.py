import json
import re
import unicodedata

CANONICAL = str.maketrans(
    {
        "\u0218": "\u015e",
        "\u0219": "\u015f",
        "\u021a": "\u0162",
        "\u021b": "\u0163",
        "\u25aa": "\u2022",
        "\u25cf": "\u2022",
        "\u2219": "\u2022",
        "\u2023": "\u2022",
        "\u2043": "\u2022",
    }
)
LEADER_RUN = re.compile(r"([._\-=·*~])\1{3,}")
QUOTE_RUN = re.compile(r"''|``")
TOKEN_PATTERN = re.compile(r"\\[A-Za-z]+|\\[{}|,;! ]|.", re.DOTALL)
MATH_SPAN = re.compile(r"\$[^$]*[\\^_{][^$]*\$")

INDIC_BLOCKS = ((0x0900, 0x097F), (0x0980, 0x09FF))
VIRAMA = {0x094D, 0x09CD}
NUKTA = {0x093C, 0x09BC}
MODIFIERS = {0x0900, 0x0901, 0x0902, 0x0903, 0x0981, 0x0982, 0x0983}


def canonical(text):
    text = unicodedata.normalize("NFC", text).translate(CANONICAL)
    text = QUOTE_RUN.sub('"', text)
    return LEADER_RUN.sub(lambda match: match.group(1) * 3, text)


def is_indic(character):
    code = ord(character)
    return any(low <= code <= high for low, high in INDIC_BLOCKS)


def indic_valid(text):
    previous = ""
    for character in text:
        if is_indic(character) and unicodedata.category(character) in ("Mn", "Mc"):
            code = ord(character)
            attached = bool(previous) and is_indic(previous)
            category = unicodedata.category(previous) if previous else ""
            if code in MODIFIERS:
                if not attached:
                    return False
            elif not (attached and (category == "Lo" or ord(previous) in NUKTA)):
                return False
        previous = character
    return True


def indic_page_ok(texts, max_bad=0.05):
    indic = [text for text in texts if any(is_indic(character) for character in text)]
    if not indic:
        return True
    return sum(not indic_valid(text) for text in indic) <= max_bad * len(indic)


class Vocab:
    def __init__(self, chars):
        self.chars = list(chars)
        self.index = {character: index + 1 for index, character in enumerate(self.chars)}
        self.has_commands = any(len(character) > 1 for character in self.chars)
        self.jamo = "\u1100" in self.index

    @classmethod
    def from_json(cls, serialized):
        data = json.loads(serialized)
        if data.get("blank", 0) != 0:
            raise ValueError("only blank=0 vocabularies are supported")
        return cls(data["chars"])

    def to_json(self):
        return json.dumps({"blank": 0, "chars": self.chars}, ensure_ascii=False)

    @property
    def n_classes(self):
        return len(self.chars) + 1

    def normalize(self, text):
        return canonical(text)

    def tokenize(self, text):
        if not self.has_commands:
            return list(text)
        out = []
        position = 0
        for match in list(MATH_SPAN.finditer(text)) + [None]:
            end = match.start() if match else len(text)
            out += self._tokens(text[position:end], False)
            if match:
                out += self._tokens(match.group(0), True)
                position = match.end()
        return out

    def _tokens(self, text, math):
        out = []
        for token in TOKEN_PATTERN.findall(text):
            if math and token.isspace():
                continue
            out.extend([token] if token in self.index or len(token) == 1 else list(token))
        return out

    def encode(self, text):
        ids = [self.index.get(token) for token in self.tokenize(self.normalize(text))]
        if not ids or any(identifier is None for identifier in ids):
            return None
        return ids

    def oov(self, text):
        return {token for token in self.tokenize(self.normalize(text)) if token not in self.index}

    def decode(self, ids):
        tokens = [self.chars[identifier - 1] for identifier in ids if 0 < identifier <= len(self.chars)]
        if self.jamo:
            return unicodedata.normalize("NFC", "".join(tokens))
        if not self.has_commands:
            return "".join(tokens)
        out = []
        for index, token in enumerate(tokens):
            out.append(token)
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            if len(token) > 2 and token[0] == "\\" and token[1:].isalpha() and following[:1].isalpha():
                out.append(" ")
        return "".join(out)
