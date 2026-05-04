import argparse
import os
import numpy as np
import scipy.io.wavfile as wavfile
import torch
from nnm.models.kitten_tts import KittenTTS


def list_voices(model):
    aliases = model.config.voice_aliases
    print('Available voices:')
    for alias, key in sorted(aliases.items()):
        print(f'  {alias:12s} -> {key}')


def main():
    parser = argparse.ArgumentParser(description='Kitten TTS Text-to-Speech')
    parser.add_argument('--model-path', type=str, required=True, help='Path to pretrained model directory')
    parser.add_argument('--text', type=str, help='Text to synthesize')
    parser.add_argument('--voice', type=str, default='Bruno', help='Voice name')
    parser.add_argument('--speed', type=float, default=None, help='Speech speed')
    parser.add_argument('--output', type=str, default='audio_output/output.wav', help='Output WAV path')
    parser.add_argument('--list-voices', action='store_true', help='List available voices')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible output')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    print('Loading model...')
    model = KittenTTS.from_pretrained(args.model_path)
    model.eval()
    model.generator.m_source.sin_gen.phase_jitter_scale = 0.0
    print('Model loaded.')

    if args.list_voices:
        list_voices(model)
        return

    if not args.text:
        parser.print_help()
        return

    speed = args.speed

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    print(f"Synthesizing: '{args.text}'")
    speed_str = f'{speed:.1f}' if speed is not None else 'default'
    print(f'Voice: {args.voice}, Speed: {speed_str}x')

    result = model.synthesize(args.text, voice=args.voice, speed=speed)
    waveform = result['waveform']
    waveform = np.clip(waveform, -1.0, 1.0)
    audio_int16 = (waveform * 32767).astype(np.int16)
    wavfile.write(args.output, result['sample_rate'], audio_int16)

    print(f'Saved to: {os.path.abspath(args.output)}')


if __name__ == '__main__':
    main()
