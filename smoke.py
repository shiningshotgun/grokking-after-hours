# No-training environment/state check: imports, versions, device, dataset
# construction, model instantiation, and one inference-only forward pass.
import torch, numpy, matplotlib

print("torch", torch.__version__)
print("numpy", numpy.__version__)
print("matplotlib", matplotlib.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

from data import build_split
from model import Transformer

train_data, val_data, tokens = build_split()
print(
    "vocab", len(tokens), "train", tuple(train_data.shape), "val", tuple(val_data.shape)
)

model = Transformer(vocab_len=len(tokens))
total = sum(p.numel() for p in model.parameters())
print("total params", total)

model.eval()
with torch.no_grad():
    logits = model(train_data[:2, :-1])
print("forward OK", tuple(logits.shape))
