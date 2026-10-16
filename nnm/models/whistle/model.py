import json
import os
import time
import wave
from dataclasses import dataclass

import numpy as np
import torch
from safetensors.torch import load_file

from ..pretrained import PretrainedModel
from .frontend import ConvStem, LogMel
from .stacks import DecoderStack, EncoderStack
from .tokenizer import WhistleTokenizer

SAMPLE_RATE = 16000
ENCODER_FRAME_MS = 80
LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")
LANGUAGE_TOKENS = {f"<|{code}|>": code for code in LANGUAGES}
MAX_AUDIO_SECONDS = 30
HADA_SLOTS = ("d1", "d2", "b2", "d3", "d4", "w1a", "w1b", "w2a", "w2b", "w3a", "w3b", "cond_v", "cond_u")
KEYWORD_BIAS = 2.0
WEIGHTS = os.path.join("checkpoints", "whistle.safetensors")
FBANK = os.path.join("checkpoints", "fbank.pt")
TOKENIZER = os.path.join("checkpoints", "tokenizer.bin")


@dataclass
class WhistleConfig:
    vocab_size: int
    d_model: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    encoder_layers: int
    qk_head_dim: int
    v_head_dim: int
    qkv_conv_taps: int
    mhc_lanes: int
    encoder_lanes: int
    hada_n: int
    rope_theta: float
    engram_slots: int
    engram_orders: tuple
    engram_layers: tuple
    engram_dilation: int
    engram_tables: int
    engram_sub_dim: int
    conv_kernel: int
    stem_channels: int
    mel_bins: int
    n_fft: int
    win_length: int = 400
    hop_length: int = 160
    sample_rate: int = SAMPLE_RATE
    max_audio_seconds: int = MAX_AUDIO_SECONDS


class Whistle(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mel = LogMel(config)
        self.stem = ConvStem(config.stem_channels, config.d_model, config.mel_bins)
        self.encoder = EncoderStack(config)
        self.decoder = DecoderStack(config)
        self.tokenizer = None

    @classmethod
    def from_pretrained(cls, pretrained_path, map_location="cpu", **kwargs):
        """A model directory holds config.json, the f32 master, the fbank and the tokenizer blob."""
        directory = cls._resolve_pretrained_path(pretrained_path, kwargs.pop("download_root", None))
        with open(os.path.join(directory, "config.json")) as handle:
            model = cls(WhistleConfig(**json.load(handle)))
        model.load_state_dict(_state_dict(load_file(os.path.join(directory, WEIGHTS))), strict=True)
        model.mel.fbank.data = torch.load(os.path.join(directory, FBANK), map_location="cpu")
        with open(os.path.join(directory, TOKENIZER), "rb") as handle:
            model.tokenizer = WhistleTokenizer(handle.read())
        return model.to(map_location).eval()

    @torch.no_grad()
    def embed(self, audio):
        return self.encoder(self.stem(self.mel(_samples(audio))))

    @torch.no_grad()
    def transcribe(self, audio, language=None, keywords=None, word_timestamps=False, beam=5, max_new_tokens=None):
        """The engine's needle_transcribe: text, language, timing, and word times when asked."""
        started = time.perf_counter()
        samples = _samples(audio)
        result = {"text": "", "language": "", "ttft_ms": 0.0, "decode_tps": 0.0}
        if word_timestamps:
            result["words"] = []
        if samples.numel() == 0 or not _has_speech(self.mel.levels(samples)):
            return result
        if max_new_tokens is None:
            max_new_tokens = min(320, 64 + int(4 * samples.numel() / self.config.sample_rate))
        memory = self.embed(samples)
        prompt = self._prompt(memory, language)
        tokens, first = self._search(memory, prompt, max_new_tokens, beam, keywords)
        result["text"] = self._text(tokens)
        result["language"] = language or self._language(tokens) or ""
        if tokens:
            result["ttft_ms"] = round((first - started) * 1000, 1)
            result["decode_tps"] = round(len(tokens) / max(time.perf_counter() - first, 1e-9), 1)
        if word_timestamps:
            result["words"] = self._align(memory, prompt, tokens, samples)
        return result

    def _prompt(self, memory, language=None):
        ids = [self.tokenizer.bos_id]
        if language:
            ids.append(self.tokenizer.piece_to_id[f"<|{language}|>"])
        return torch.tensor([ids], dtype=torch.long, device=memory.device)

    def _logprobs(self, tokens, memory):
        memory = memory.unsqueeze(0) if memory.dim() == 2 else memory
        logits = self.decoder(tokens, memory=memory.expand(tokens.shape[0], -1, -1))
        return torch.log_softmax(logits[:, -1], dim=-1)

    def _search(self, memory, prompt, max_new_tokens, beam, keywords=None):
        return (
            self._beam(memory, prompt, max_new_tokens, beam, keywords)
            if beam > 1
            else self._greedy(memory, prompt, max_new_tokens, keywords)
        )

    def _greedy(self, memory, prompt, max_new_tokens, keywords=None):
        trie = KeywordTrie(keywords, self.tokenizer)
        tokens, node, generated, first = prompt, trie.root, [], None
        for _ in range(max_new_tokens):
            logprobs = trie.boost(self._logprobs(tokens, memory), [node])[0]
            first = first or time.perf_counter()
            token = int(logprobs.argmax())
            if token == self.tokenizer.eos_id:
                break
            generated.append(token)
            node = trie.advance(node, token)
            tokens = torch.cat([tokens, torch.tensor([[token]], device=tokens.device)], dim=1)
        return generated, first or time.perf_counter()

    def _beam(self, memory, prompt, max_new_tokens, width, keywords=None):
        def rank(candidate):
            return candidate[0] / ((5 + len(candidate[1])) / 6)

        trie = KeywordTrie(keywords, self.tokenizer)
        beams, finished, first = [(0.0, [], trie.root)], [], None
        for _ in range(max_new_tokens):
            if not beams:
                break
            history = torch.tensor([beam[1] for beam in beams], dtype=torch.long, device=prompt.device)
            tokens = torch.cat([prompt.expand(len(beams), -1), history], dim=1)
            logprobs = trie.boost(self._logprobs(tokens, memory), [beam[2] for beam in beams])
            first = first or time.perf_counter()
            top = logprobs.topk(width, dim=-1)
            ranked = sorted(
                (
                    (
                        score + float(top.values[index, rank_]),
                        beam + [int(top.indices[index, rank_])],
                        trie.advance(node, int(top.indices[index, rank_])),
                    )
                    for index, (score, beam, node) in enumerate(beams)
                    for rank_ in range(width)
                ),
                key=lambda candidate: -rank(candidate),
            )
            finished.extend((score, beam[:-1], node) for score, beam, node in ranked if beam[-1] == self.tokenizer.eos_id)
            beams = [candidate for candidate in ranked if candidate[1][-1] != self.tokenizer.eos_id][:width]
        best = max(finished + beams, key=rank) if finished or beams else (0.0, [], trie.root)
        return best[1], first or time.perf_counter()

    def _align(self, memory, prompt, tokens, samples):
        """Word spans: a monotonic partition of the encoder frames by cross-attention."""
        if not tokens:
            return []
        full = torch.cat([prompt, torch.tensor([tokens], dtype=torch.long, device=prompt.device)], dim=1)
        logits = self.decoder(full, memory=memory.unsqueeze(0))
        logprobs = torch.log_softmax(logits[0], dim=-1)
        attention = torch.stack([block.cross.attention[0].mean(0) for block in self.decoder.blocks]).mean(0)
        pieces, tokenizer = self.tokenizer.pieces, self.tokenizer
        row = {token: prompt.shape[1] - 1 + token for token in range(len(tokens))}
        rows = [token for token in range(len(tokens)) if _is_content(tokens[token], tokenizer)]
        edges = _monotonic_spans(attention[[row[token] for token in rows]], len(rows))
        slots = {token: slot for slot, token in enumerate(rows)}
        step = ENCODER_FRAME_MS / 1000
        duration = samples.numel() / self.config.sample_rate
        words, current = [], None
        for token in range(len(tokens)):
            piece = pieces[tokens[token]]
            if piece in LANGUAGE_TOKENS:
                continue
            if current is not None and piece.startswith("\u2581"):
                words.append(current)
                current = None
            current = current or {"ids": [], "frames": [], "probability": []}
            current["ids"].append(tokens[token])
            if (slot := slots.get(token)) is not None:
                current["frames"] += [edges[slot], edges[slot + 1]]
                current["probability"].append(float(logprobs[row[token], tokens[token]].exp()))
        words += [current] if current is not None else []
        for word in words:
            word["word"] = self.tokenizer.decode(word["ids"]).strip()
            if not word["frames"]:
                word["start"] = word["end"] = word["probability"] = 0.0
                continue
            word["start"] = round(min(word["frames"]) * step, 2)
            word["end"] = min(round(max(word["frames"]) * step, 2), int(duration * 100) / 100)
            word["probability"] = round(float(np.prod(word["probability"]) ** (1 / len(word["probability"]))), 3)
        return words

    @torch.no_grad()
    def stream(self, chunks, language=None, keywords=None, beam=5):
        """Live transcription: one dict per chunk, then one for the tail when it ends.

        Each dict carries the "text" and "words" committed by that chunk, the
        unconfirmed "pending" tail, the "language", the seconds "received" and the
        "pass_ms" it took. A word is committed once it ends before the previous
        chunk, so about one chunk of audio stays pending; the tail is flushed on
        the final call, which the engine signals with an empty chunk.
        """
        rate = self.config.sample_rate
        buffered, committed, detected = [], [], language
        received = 0.0
        for chunk in list(chunks) + [torch.zeros(0)]:
            samples = _samples(chunk)
            started = time.perf_counter()
            tail = samples.numel() == 0
            if not tail:
                buffered.append(samples)
                received += samples.numel() / rate
            audio = torch.cat(buffered) if buffered else torch.zeros(1)
            window = audio[-self.config.max_audio_seconds * rate :]
            offset = (audio.numel() - window.numel()) / rate
            tokens, words = self._window(window, beam, language, keywords)
            detected = detected or language or self._language(tokens) or ""
            for word in words:
                word["start"] += offset
                word["end"] += offset
            cut = float("inf") if tail else received - samples.numel() / rate
            overlap = 0
            while overlap < len(committed) and overlap < len(words) and words[overlap]["word"] == committed[overlap]["word"]:
                overlap += 1
            fresh = [word for word in words[overlap:] if word["end"] <= cut]
            committed = words[: overlap + len(fresh)]
            pending = words[overlap + len(fresh) :]
            yield {
                "text": " ".join(word["word"] for word in fresh).strip(),
                "words": fresh,
                "pending": " ".join(word["word"] for word in pending).strip(),
                "language": detected,
                "received": received,
                "pass_ms": round((time.perf_counter() - started) * 1000, 1),
            }

    def _window(self, audio, beam, language, keywords):
        """Decode one window of the stream and align its words to the frames."""
        if audio.numel() == 0 or not _has_speech(self.mel.levels(audio)):
            return [], []
        memory = self.embed(audio)
        prompt = self._prompt(memory, language)
        limit = min(320, 64 + int(4 * audio.numel() / self.config.sample_rate))
        tokens, _ = self._search(memory, prompt, limit, beam, keywords)
        return tokens, self._align(memory, prompt, tokens, audio)

    def _language(self, tokens):
        for token in tokens:
            code = LANGUAGE_TOKENS.get(self.tokenizer.pieces[token])
            if code is not None:
                return code
        return None

    def _text(self, tokens):
        for index, token in enumerate(tokens):
            if self.tokenizer.pieces[token] in LANGUAGE_TOKENS:
                return self.tokenizer.decode(tokens[index + 1 :])
        return self.tokenizer.decode(tokens)


def _state_dict(tensors):
    """The master stores one batched tensor per layer under JAX names; split it into module weights."""
    def rows(key):
        return torch.as_tensor(tensors[key])

    def kernel(key, layer):
        return rows(key)[layer].T

    def channels(prefix, layer, count):
        return {f"{prefix}.{layer}.{tail}.weight": kernel(f"{head}/kernel", layer)
                for tail, head in count}

    def norms(prefix, layer, count):
        return {f"{prefix}.{layer}.{tail}.scale": rows(f"{head}/scale")[layer]
                for tail, head in count}

    def taps(prefix, layer):
        head = "stack/layers/block/self_attn"
        return {f"{prefix}.{layer}.self_attn.{name}": rows(f"{head}/{name}")[layer]
                for name in ("q_taps", "k_taps", "v_taps")}

    state = {"decoder.embedding.weight": rows("embedding/embedding")}
    state["decoder.final_norm.scale"] = rows("stack/final_norm/scale")
    state["encoder.final_norm.scale"] = rows("encoder/final_norm/scale")
    state |= {f"decoder.engrams.{index}.embedding": rows(f"engrams_{index}/embedding")
              for index in range(sum(key.endswith("/key_proj/kernel") for key in tensors))}
    for index in range(sum(key.endswith("/key_proj/kernel") for key in tensors)):
        state[f"decoder.engrams.{index}.key_proj.weight"] = rows(f"engrams_{index}/key_proj/kernel").T
        state[f"decoder.engrams.{index}.value_proj.weight"] = rows(f"engrams_{index}/value_proj/kernel").T
        state[f"decoder.engrams.{index}.taps"] = rows(f"engrams_{index}/taps")
    state |= {
        f"decoder.mhc.{name}": rows(f"stack/mhc_{name}")
        for name in ("a_pre", "a_post", "a_res", "b_pre", "b_post", "b_res")
    }
    state |= {f"decoder.mhc.phi_{name}": rows(f"stack/mhc_phi_{name}").transpose(-1, -2)
              for name in ("pre", "post", "res")}
    state["encoder.pe_gate"] = rows("pe_gate").reshape(())
    state |= {f"encoder.mhc.{name}": rows(f"encoder/mhc_{name}")
              for name in ("a_pre", "a_post", "a_res", "b_pre", "b_post", "b_res")}
    state |= {f"encoder.mhc.phi_{name}": rows(f"encoder/mhc_phi_{name}").transpose(-1, -2)
              for name in ("pre", "post", "res")}
    for tail, head in (("stem.conv", "stem/w"), ("stem.dw_1", "stem/dw_1"), ("stem.dw_2", "stem/dw_2")):
        state[f"{tail}.weight"] = rows(head).permute(3, 2, 0, 1)
    for tail, head in (("stem.pw_1", "stem/pw_1"), ("stem.pw_2", "stem/pw_2"), ("stem.out", "stem/out")):
        state[f"{tail}.weight"] = rows(f"{head}/kernel").T
    for layer in range(rows("stack/layers/block/attn_gate").shape[0]):
        state |= channels(f"decoder.blocks", layer, (
            ("self_attn.q_proj", "stack/layers/block/self_attn/q_proj"),
            ("self_attn.k_proj", "stack/layers/block/self_attn/k_proj"),
            ("self_attn.v_proj", "stack/layers/block/self_attn/v_proj"),
            ("self_attn.out_proj", "stack/layers/block/self_attn/out_proj"),
            ("self_attn.gate_proj", "stack/layers/block/self_attn/gate_proj"),
            ("cross.q_proj", "stack/layers/block/cross_attn/q_proj"),
            ("cross.k_proj", "stack/layers/block/cross_attn/k_proj"),
            ("cross.v_proj", "stack/layers/block/cross_attn/v_proj"),
            ("cross.out_proj", "stack/layers/block/cross_attn/out_proj"),
            ("cross.gate_proj", "stack/layers/block/cross_attn/gate_proj"),
        ))
        state |= norms("decoder.blocks", layer, (
            ("norm", "stack/layers/block/ZCRMSNorm_0"),
            ("post_attn_norm", "stack/layers/block/post_attn_norm"),
            ("cross_norm", "stack/layers/block/cross_norm"),
            ("post_cross_norm", "stack/layers/block/post_cross_norm"),
            ("pre_hada_norm", "stack/layers/block/pre_hada_norm"),
            ("self_attn.q_norm", "stack/layers/block/self_attn/q_norm"),
            ("self_attn.k_norm", "stack/layers/block/self_attn/k_norm"),
            ("cross.q_norm", "stack/layers/block/cross_attn/q_norm"),
            ("cross.k_norm", "stack/layers/block/cross_attn/k_norm"),
        ))
        state |= taps("decoder.blocks", layer)
        state[f"decoder.blocks.{layer}.attn_gate"] = rows("stack/layers/block/attn_gate")[layer]
        state[f"decoder.blocks.{layer}.cross_gate"] = rows("stack/layers/block/cross_gate")[layer]
        state |= {f"decoder.blocks.{layer}.mlp.{name}": rows(f"stack/layers/block/hadamard_mlp/{name}")[layer]
                  for name in HADA_SLOTS}
    for layer in range(rows("encoder/layers/block/attn_gate").shape[0]):
        state |= channels("encoder.blocks", layer, (
            ("attn.q_proj", "encoder/layers/block/self_attn/q_proj"),
            ("attn.k_proj", "encoder/layers/block/self_attn/k_proj"),
            ("attn.v_proj", "encoder/layers/block/self_attn/v_proj"),
            ("attn.out_proj", "encoder/layers/block/self_attn/out_proj"),
            ("attn.gate_proj", "encoder/layers/block/self_attn/gate_proj"),
            ("conv.pw1", "encoder/layers/block/pw1"),
            ("conv.pw2", "encoder/layers/block/pw2"),
        ))
        state |= norms("encoder.blocks", layer, (
            ("norm", "encoder/layers/block/ZCRMSNorm_0"),
            ("post_attn_norm", "encoder/layers/block/post_attn_norm"),
            ("conv_norm", "encoder/layers/block/conv_norm"),
            ("conv_out_norm", "encoder/layers/block/conv_out_norm"),
            ("pre_hada_norm", "encoder/layers/block/pre_hada_norm"),
            ("pre_hada_norm_0", "encoder/layers/block/pre_hada_norm_0"),
            ("attn.q_norm", "encoder/layers/block/self_attn/q_norm"),
            ("attn.k_norm", "encoder/layers/block/self_attn/k_norm"),
        ))
        state[f"encoder.blocks.{layer}.attn_gate"] = rows("encoder/layers/block/attn_gate")[layer]
        state[f"encoder.blocks.{layer}.conv.dw.weight"] = rows("encoder/layers/block/dw")[layer].permute(2, 1, 0)
        for kind, head in (("mlp", "encoder/layers/block/hadamard_mlp"), ("mlp0", "encoder/layers/block/hadamard_mlp_0")):
            state |= {f"encoder.blocks.{layer}.{kind}.{name}": rows(f"{head}/{name}")[layer] for name in HADA_SLOTS}
    return state


def _is_content(token, tokenizer):
    piece = tokenizer.pieces[token]
    if token in (tokenizer.bos_id, tokenizer.eos_id) or piece in LANGUAGE_TOKENS:
        return False
    return any(character.isalnum() for character in piece)


def _monotonic_spans(attention, count):
    """Partition every encoder frame among `count` rows, left to right, maxing attention mass."""
    scores = attention.numpy().astype(np.float64)
    frames = scores.shape[1]
    if count == 0 or frames < count:
        return []
    cumulative = np.concatenate([np.zeros((count, 1)), np.cumsum(scores, axis=1)], axis=1)
    best = np.full((count + 1, frames + 1), -1e9)
    cuts = np.zeros((count + 1, frames + 1), dtype=int)
    best[0, 0] = 0.0
    for row in range(1, count + 1):
        for end in range(row, frames - (count - row) + 1):
            starts = np.arange(row - 1, end)
            totals = best[row - 1, starts] + cumulative[row - 1, end] - cumulative[row - 1, starts]
            pick = int(totals.argmax())
            best[row, end] = totals[pick]
            cuts[row, end] = starts[pick]
    edges, end = [frames], frames
    for row in range(count, 0, -1):
        end = int(cuts[row, end])
        edges.append(end)
    return list(reversed(edges))


class KeywordTrie:
    """The engine's keyword biasing: a token trie that boosts the words it was asked to favour."""

    END = -1

    def __init__(self, keywords, tokenizer):
        self.root = {}
        for keyword in keywords or []:
            for text in (keyword, " " + keyword):
                node = self.root
                for token in tokenizer.encode(text):
                    node = node.setdefault(token, {})
                node[self.END] = {}

    def boost(self, logprobs, nodes, amount=None):
        amount = KEYWORD_BIAS if amount is None else amount
        for index, node in enumerate(nodes):
            for token in node:
                if token != self.END:
                    logprobs[index, token] += amount
        return logprobs

    def advance(self, node, token):
        if (child := node.get(token)) is not None:
            return child
        return self.root.get(token, self.root)


def _has_speech(levels, threshold=5.0):
    """The engine's gate: P99 - P10 of the per-frame level has to clear 5 dB."""
    ordered = levels.sort().values
    count = ordered.numel()
    return float(ordered[int((count - 1) * 0.99)] - ordered[int((count - 1) * 0.1)]) >= threshold


def _samples(audio):
    if isinstance(audio, (str, os.PathLike)):
        return _read_wav(os.fspath(audio))
    if isinstance(audio, torch.Tensor):
        return audio.to(torch.float32).reshape(-1)
    return torch.from_numpy(np.asarray(audio, dtype=np.float32).reshape(-1))


def _read_wav(path):
    with wave.open(path, "rb") as source:
        if source.getsampwidth() != 2 or source.getcomptype() != "NONE":
            raise ValueError(f"{path} must be 16-bit PCM")
        raw, channels, rate = source.readframes(source.getnframes()), source.getnchannels(), source.getframerate()
    values = np.frombuffer(raw, np.int16).astype(np.float32) / (1 << 15)
    if channels > 1:
        values = values.reshape(-1, channels).mean(1)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path} must be {SAMPLE_RATE} Hz")
    return torch.from_numpy(values)
