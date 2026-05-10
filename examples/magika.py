import argparse
import torch
from nnm.models.magika import Magika


@torch.no_grad()
def predict(model, raw_bytes):
    x = torch.tensor(bytearray(raw_bytes), dtype=torch.uint8).reshape(1, -1)
    probs = model(x).numpy().flatten()
    pred_idx = int(probs.argmax())
    return model.labels[pred_idx], probs[pred_idx]


def main():
    parser = argparse.ArgumentParser(description='File type detection with nnm.magika')
    parser.add_argument('--model-path', type=str, required=True, help='Path to pretrained model directory')
    parser.add_argument('--file', type=str, required=True, help='File path to identify')
    args = parser.parse_args()

    model = Magika.from_pretrained(args.model_path)
    model.eval()

    with open(args.file, 'rb') as f:
        raw_bytes = f.read()
    label, score = predict(model, raw_bytes)
    print(f'{args.file}: {label} (score={score:.4f})')


if __name__ == '__main__':
    main()