import argparse
import json
import os

import numpy as np
import scipy.io.wavfile as wavfile
import torch

from nnm.models.kitten_tts import KittenTTS, tokens_from_phonemes


def main():
    parser = argparse.ArgumentParser(description="Kitten TTS 2 (nano/pico/mini) neural inference")
    parser.add_argument("--model-path", required=True, help="nnm weight repo directory")
    parser.add_argument("--variant", default=None, help="language variant (nano/pico), e.g. en, de")
    parser.add_argument("--voice", default=None, help="voice key from voices_<lang>.npz; default: first")
    parser.add_argument("--phonemes", default="həlˈoʊ wˈɜːld", help="phoneme string to synthesize")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--output", default="audio_output/output.wav")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    model = KittenTTS.from_pretrained(args.model_path, variant=args.variant).eval()
    directory = args.model_path
    vocab = json.load(open(os.path.join(directory, "vocab.json"), encoding="utf-8"))
    lang = args.variant or "en"
    packs = np.load(os.path.join(directory, f"voices_{lang}.npz"))
    voice = args.voice or packs.files[0]
    pack = packs[voice]

    tokens = tokens_from_phonemes(args.phonemes, vocab)
    style = torch.from_numpy(pack[min(len(tokens) - 1, pack.shape[0] - 1)].copy()).float().unsqueeze(0)
    input_ids = torch.tensor([tokens], dtype=torch.long)

    result = model.infer(input_ids, style, speed=args.speed)
    waveform = np.clip(result["waveform"].numpy(), -1.0, 1.0)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    wavfile.write(args.output, model.config.sample_rate, (waveform * 32767).astype(np.int16))
    print(f"Saved {os.path.abspath(args.output)} ({len(waveform)} samples, voice={voice})")


if __name__ == "__main__":
    main()
