#!/usr/bin/env python3
import math

import torch

from tiny_genius.checkpoint import load_checkpoint
from tiny_genius.config import REPO_ROOT
from tiny_genius.model import TinyModelConfig, TinyTransformer
from tiny_genius.tokenizer import Tokenizer
from tiny_genius.tokenizer.specials import PAD_ID

OUT = REPO_ROOT / "artifacts" / "stage4_300m_learning"
PACKED = OUT / "packed_sequences.pt"
CKPT = OUT / "checkpoints" / "final.pt"

def loss(model, batch):
    with torch.no_grad():
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=PAD_ID,
        ).item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = Tokenizer.load_frozen()
    cfg = TinyModelConfig.from_yaml(REPO_ROOT / "configs" / "model_300m.yaml")
    model = TinyTransformer(cfg).to(device)

    payload = load_checkpoint(CKPT)
    state = payload.get("model", payload.get("model_state_dict"))
    if state is None:
        raise RuntimeError(f"Could not find model weights in checkpoint: {payload.keys()}")

    model.load_state_dict(state)
    model.eval()

    packed = torch.load(PACKED, map_location="cpu", weights_only=False)
    sequences = packed["sequences"]

    # Same convention as the training script: held-out tail.
    split = int(sequences.shape[0] * 0.9)
    eval_seq = sequences[split:]

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(eval_seq.shape[0]):
            batch = eval_seq[i:i+1].to(device)
            step_loss = loss(model, batch)
            n = int((batch[:, 1:] != PAD_ID).sum())
            total_loss += step_loss * n
            total_tokens += n

    avg = total_loss / total_tokens
    print(f"device:       {device}")
    print(f"eval seqs:    {len(eval_seq)}")
    print(f"eval tokens:  {total_tokens:,}")
    print(f"eval loss:    {avg:.6f}")
    print(f"eval ppl:     {math.exp(avg):.3f}")
    print(f"tokenizer:    {tok.fingerprint}")
    print(f"checkpoint:   {CKPT}")
    print("training:     NO")

if __name__ == "__main__":
    main()
