import os
import re

import espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper

EspeakWrapper.set_library(espeakng_loader.get_library_path())
os.environ['ESPEAK_DATA_PATH'] = espeakng_loader.get_data_path()

import phonemizer

_PHONEMIZER = None
_CLEANER = None


def _get_phonemizer():
    global _PHONEMIZER
    if _PHONEMIZER is None:
        _PHONEMIZER = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
    return _PHONEMIZER


def _get_cleaner():
    global _CLEANER
    if _CLEANER is None:
        _CLEANER = TextCleaner()
    return _CLEANER


class TextCleaner:
    _PAD = "$"
    _PUNCTUATION = ';:,.!?¡¿—…"«»"" '
    _LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    _LETTERS_IPA = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

    def __init__(self):
        symbols = [self._PAD] + list(self._PUNCTUATION) + list(self._LETTERS) + list(self._LETTERS_IPA)
        self.word_index_dictionary = {s: i for i, s in enumerate(symbols)}

    def __call__(self, text):
        indexes = []
        for char in text:
            idx = self.word_index_dictionary.get(char)
            if idx is not None:
                indexes.append(idx)
        return indexes


def basic_english_tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)


def ensure_punctuation(text):
    text = text.strip()
    if not text:
        return text
    if text[-1] not in '.!?,;:':
        text = text + ','
    return text


def phonemize(text):
    phonemizer_instance = _get_phonemizer()
    result = phonemizer_instance.phonemize([text])
    return result[0]


def _tokens_from_phonemes_str(phonemes_str):
    phonemes_list = basic_english_tokenize(phonemes_str)
    joined = ' '.join(phonemes_list)
    cleaner = _get_cleaner()
    tokens = cleaner(joined)
    tokens.insert(0, 0)
    tokens.append(10)
    tokens.append(0)
    return tokens


def text_to_tokens(text):
    text = ensure_punctuation(text)
    phonemes = phonemize(text)
    return _tokens_from_phonemes_str(phonemes)


def phonemes_to_tokens(phonemes_str):
    return _tokens_from_phonemes_str(phonemes_str)
